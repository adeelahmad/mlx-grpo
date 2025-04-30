# reward.py
import math
import logging
import re
import string
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Set, Any  # Added Any
from rouge import Rouge
from nltk.translate import bleu_score
from nltk.translate.bleu_score import SmoothingFunction

try:
    import numpy as np
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    # Ensure NLTK data is available
    try:
        import nltk

        nltk.data.find("tokenizers/punkt")
        nltk.data.find("corpora/stopwords")
        from nltk.corpus import stopwords

        STOPWORDS = set(stopwords.words("english"))
    except LookupError:
        logging.warning("NLTK stopwords or punkt tokenizer not found. Downloading...")
        try:
            nltk.download("punkt", quiet=True)
            nltk.download("stopwords", quiet=True)
            from nltk.corpus import stopwords

            STOPWORDS = set(stopwords.words("english"))
        except Exception as nltk_e:
            logging.error(
                f"Failed to download NLTK data: {nltk_e}. Using basic hardcoded STOPWORDS."
            )
            STOPWORDS = set(
                [
                    "i",
                    "me",
                    "my",
                    "myself",
                    "we",
                    "our",
                    "ours",
                    "ourselves",
                    "you",
                    "your",
                    "yours",
                    "yourself",
                    "yourselves",
                    "he",
                    "him",
                    "his",
                    "himself",
                    "she",
                    "her",
                    "hers",
                    "herself",
                    "it",
                    "its",
                    "itself",
                    "they",
                    "them",
                    "their",
                    "theirs",
                    "themselves",
                    "what",
                    "which",
                    "who",
                    "whom",
                    "this",
                    "that",
                    "these",
                    "those",
                    "am",
                    "is",
                    "are",
                    "was",
                    "were",
                    "be",
                    "been",
                    "being",
                    "have",
                    "has",
                    "had",
                    "having",
                    "do",
                    "does",
                    "did",
                    "doing",
                    "a",
                    "an",
                    "the",
                    "and",
                    "but",
                    "if",
                    "or",
                    "because",
                    "as",
                    "until",
                    "while",
                    "of",
                    "at",
                    "by",
                    "for",
                    "with",
                    "about",
                    "against",
                    "between",
                    "into",
                    "through",
                    "during",
                    "before",
                    "after",
                    "above",
                    "below",
                    "to",
                    "from",
                    "up",
                    "down",
                    "in",
                    "out",
                    "on",
                    "off",
                    "over",
                    "under",
                    "again",
                    "further",
                    "then",
                    "once",
                    "here",
                    "there",
                    "when",
                    "where",
                    "why",
                    "how",
                    "all",
                    "any",
                    "both",
                    "each",
                    "few",
                    "more",
                    "most",
                    "other",
                    "some",
                    "such",
                    "no",
                    "nor",
                    "not",
                    "only",
                    "own",
                    "same",
                    "so",
                    "than",
                    "too",
                    "very",
                    "s",
                    "t",
                    "can",
                    "will",
                    "just",
                    "don",
                    "should",
                    "now",
                ]
            )

except ImportError as import_err:
    logging.error(
        f"Core dependency missing: {import_err}. Reward function requires numpy, scikit-learn, rouge-score, and nltk."
    )
    raise ImportError("Missing core reward dependencies.") from import_err


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)



@dataclass
class RewardConfig:
    """Configuration for the SimpleRewardFunction."""

    # --- Weights (Ensure sum ~1.0) ---
    # **ADJUSTED Weights (Example - Tune based on impact)**
    reward_overall_format_weight: float = 0.10
    reward_tag_presence_weight: float = 0.05
    think_non_repetition_weight: float = 0.05 # Slightly decreased from 0.10
    think_conciseness_weight: float = 0.10
    think_positive_words_weight: float = 0.10
    think_negative_words_weight: float = 0.05
    think_critical_negative_words_weight: float = 0.10 # Decreased from 0.15
    answer_accuracy_weight: float = 0.45 # Increased from 0.35
    # Note: Weights now sum to ~1.0. Re-verify after tuning.

    # --- Thresholds & Targets ---
    non_repetition_threshold: float = 0.8
    non_repetition_history_length: int = 10
    non_repetition_min_step_words: int = 3
    non_repetition_compare_depth: int = 5
    conciseness_target_tokens: int = 100
    conciseness_tolerance_ratio: float = 0.30
    max_positive_words_count: int = 25
    max_negative_words_excess: int = 5

    # --- Normalization ---
    reward_ema_alpha: float = 0.05
    # **CRITICAL FIX: ADJUSTED Reward Clipping**
    min_reward_clip: float = -1.5 # Adjusted based on negative Rew(orig) logs
    max_reward_clip: float = 1.5 # Adjusted to allow some positive reward

    # --- Word Lists ---
    # (Keeping lists as provided, review/tune these if reward logic still seems off)
    positive_words: List[str] = field(
        default_factory=lambda: [
            "User is asking", "Okay, I am thinking about the request", "The user provided ",
            "I must focus on the core objective of the request", "Goal of User request is", "I ", "Step ", "remember ",
            "Let", "Assuming ", "Okay ", "Next,", "Another ", "I'll", "The user wants me ", "The core task is to ",
            "The user specifically mentioned", "the user wants the final output ", "The user emphasized ",
            "Okay, I can help with that", "I also", "I am being asked", "I must provide", "it aligns with User's Reqyest",
            "Here's a breakdown of the thinking process ", "Understand the Goal", "the primary goal is", "Let's apply ",
            "Now that i have verified", "I remember ", "I read about", "lets validate each step", "let me verify again ",
            "Given the context ", "Lets recalculate ", "But wait, the user said ", "So that means ", "I am thinking about ",
            "User request is about ", "Lets take a step back ", "Lets track this constraint ", "Hypothesis ",
            "Do not want to overwhelm the user ", "implicit constraints ", "explicit constraint ", "logical gaps",
            "I am considering approach ", "I am leaning towards", "let me ask user ", "notice ", "Looking ", "First,",
            "We ", "I'm ", "Let's ", "break ", "thinking ", "breakdown ", "should ", "plan ", "imagine", "edge",
            "scenarios", "carefully", "thinking", "I am", "detailed", "Okay!\n I", "Therefore,", "This means ",
            "It follows that", "This implies", "Consequently,", "I can conclude", "This confirms ", "Given this constraint",
            "This is consistent with", "Looking back at", "Validating this", "Double-checking ", "To verify ",
            "This aligns with", "Upon reflection", "Constraint:", "Rule:", "Tracking constraints:", "Given that",
            "Updating my understanding", "Combining constraints", "Revisiting the constraint", "Constraint satisfaction check",
            "Ensuring consistency with", "This satisfies", "All constraints now satisfied", "Case analysis:",
            "Testing hypothesis:", "Eliminating possibilities", "Systematic approach", "Working backward",
            "Process of elimination", "Cross-referencing", "Deductive reasoning", "If-then analysis",
            "Chain of implications", "Logical deduction", "Direct inference", "Verification step:", "Self-check:",
            "Consistency verification:", "Validating solution:", "Testing answer against rules:", "Final verification:",
            "Cross-checking solution", "Re-examining the solution",
        ] +  [
            "Okay, I am figuring out",
            "Okay, let me think about ",
            "Okay, I am being asking...",
            "Alright, processing this...",
            "Okay, thinking through the options for",
            "Let's see, considering",
            "Hmm, let me evaluate",
            "Working on figuring out",
            "Just a moment while I process",
            "Okay, contemplating",
            "Thinking step-by-step about",
            "Alright, let me analyze",
            "Okay, gathering my thoughts on",
            "Let me just consider",
            "Okay, running through the possibilities for",
            "Mulling over",
            "Okay, assessing",
            "One sec, thinking about",
            "Right, let me structure my thoughts on",
            "Okay, formulating a response regarding",
            "Let me check on",
            "Okay, working on",
            "Okay, processing that request...",
            "Okay, thinking through the options now.",
            "Okay, let me analyze this.",
            "Okay, considering the best approach.",
            "Okay, gathering my thoughts.",
            "Okay, working on structuring that.",
            "Okay, let me check the details.",
            "Okay, formulating a response.",
            "Okay, assessing the information.",
            "Okay, planning the steps.",
            "Okay, running through the possibilities.",
            "Okay, let me review that.",
            "Okay, breaking that down.",
            "Okay, need a moment to think.",
            "Okay, evaluating the parameters.",
            "Okay, getting the context right.",
            "Okay, just processing...",
            "Let me think about that for a second.",
            "Let's see, considering the angles.",
            "Let me analyze the components.",
            "Let's see how to best explain this.",
            "Let me work through this logic.",
            "Let's see, what are the key points?",
            "Let me evaluate the possibilities.",
            "Let's consider the context here.",
            "Let me figure out the structure.",
            "Let's organize these thoughts.",
            "Let me process this information.",
            "Let's see about the best way forward.",
            "Let me double-check that.",
            "Let's refine this idea.",
            "Let me make sure I understand correctly.",
            "Hmm, let me think...",
            "Hmm, analyzing the request.",
            "Hmm, considering the nuances.",
            "Hmm, how to put this...",
            "Hmm, working out the details.",
            "Thinking about the best way to respond.",
            "Thinking through the implications.",
            "Thinking step-by-step.",
            "Considering the request carefully.",
            "Considering all factors.",
            "Contemplating the question.",
            "Mulling over the possibilities.",
            "Processing the information now...",
            "Processing...",
            "Analyzing the query...",
            "Analyzing the details provided.",
            "Evaluating the options.",
            "Evaluating the context.",
            "Figuring out the best path.",
            "Figuring this out...",
            "Synthesizing the information.",
            "Breaking down the problem.",
            "Working on it now.",
            "Working on formulating an answer.",
            "Working out the logic.",
            "Structuring my thoughts.",
            "Structuring the response.",
            "Planning the explanation.",
            "Outlining the key points.",
            "Organizing the information.",
            "Preparing the details.",
            "Drafting a response structure.",
            "Just a moment while I process this.",
            "Just a moment, thinking...",
            "One sec, let me think.",
            "One moment, evaluating.",
            "Hold on, processing request.",
            "Allow me a moment to consider.",
            "Pausing to think.",
            "Taking a second to structure this.",
            "Briefly considering...",
            "Alright, let me analyze that.",
            "Alright, processing this...",
            "Alright, thinking about how to proceed.",
            "Alright, let's break this down.",
            "Right, let me structure my thoughts on",
            "Right, considering the best response.",
            "Right, working on that now.",
            "Checking on the details.",
            "Reviewing the context.",
            "Assessing the situation.",
            "Determining the best course.",
            "Understanding the core request.",
            "Exploring the possibilities.",
            "Investigating the query.",
            "Deliberating on the best answer.",
            "Getting clarification internally.",
            "Cross-referencing information.",
            "Initiating thought process.",
            "Compiling relevant data points.",
            "Mapping out the response.",
            "Reflecting on the question.",
            "Examining the request.",
            "Preparing the answer structure.",
        ]
    )
    negative_words: List[str] = field(
        default_factory=lambda: [
            "Confused", "frustrated", "frustrating", "Alternatively", "Here's", "Chain of Thoughts,", "Another option",
            "Another way", "On the other hand", "Actually,", "Probably", "Maybe", "Might be", "Could be", "Possibly",
            "Not sure", "Unclear", "I'm guessing", "It seems", "Likely", "Perhaps", "Going back to", "As I stated earlier",
            "Repeating myself", "Circular reasoning", "We've been here before", "Revisiting the same point",
            "The reasoning structure seems unclear or incomplete", "Try outlining steps or thinking step-by-step.",
            "Another choice", "Somehow", "In some way", "Magically", "Let's just say", "For some reason", "moving on",
            "Skipping ahead", "Without going into details", "Too complicated to explain", "It just works",
            "Step-by-Step Reasoning**", "Another", "```","CoT Guidance","[another way]",
        ]
    )
    critical_negative_words: List[str] = field(
        default_factory=lambda: [
            "Let me rethink", "No, that's wrong", "Correction", "I made a mistake", "That doesn't make sense",
            "Wait, that can't be right", "That's impossible", "This doesn't add up", "Something is off", "CoT Guidance:",
            "Constraint violated", "Rule broken", "Inconsistent with", "Failed check", "Contradiction found",
            "False assumption detected", "This is invalid", "Another way", "However, thats a contradiction",
            "Thats a contractdiction","Another way","[Another way]()","COT","Chain of Thoughts","Reasoning Structure",
        ]
    )

    # --- Special Tokens & Placeholders ---
    special_tokens: Dict[str, str] = field(
        default_factory=lambda: {
            "think_start": "<thinking>",
            "think_end": "</thinking>",
            "answer_start": "<answer>",
            "answer_end": "</answer>",
            "sep": "\n",
        }
    )
    min_placeholder_reward_text: str = "<thinking>\n(Minimum valid thinking content)\n</thinking>\n<answer>\n(Minimum valid answer content)\n</answer>"

    def __post_init__(self):
        """Initialize default word lists and validate weights."""
        total_weight = sum(
            [
                self.reward_overall_format_weight,
                self.reward_tag_presence_weight,
                self.think_non_repetition_weight,
                self.think_conciseness_weight,
                self.think_positive_words_weight,
                self.think_negative_words_weight,
                self.think_critical_negative_words_weight,
                self.answer_accuracy_weight,
            ]
        )
        # Use a slightly tighter tolerance if manually adjusting weights to sum exactly
        if not math.isclose(total_weight, 1.0, abs_tol=0.01):
            logging.warning(
                f"RewardConfig weights sum to {total_weight:.3f}, deviates significantly from 1.0. Normalization might be affected."
            )
        if not isinstance(self.positive_words, list): self.positive_words = []
        if not isinstance(self.negative_words, list): self.negative_words = []
        if not isinstance(self.critical_negative_words, list): self.critical_negative_words = []


class SimpleRewardFunction:
    """
    Calculates rewards for generated text based on format, accuracy, conciseness,
    repetition, tag presence, and use of specific words, applying metrics to thinking
    and answer blocks separately. Includes enhanced negative word penalties.
    """

    def __init__(self, config: Optional[RewardConfig] = None, verbose: bool = False):
        self.config = config if config else RewardConfig()
        self.verbose = verbose

        # State for non-repetition check and reward normalization
        self.previous_reasoning_steps: List[str] = []
        self.reward_ema: float = 0.0
        self.reward_std: float = 1.0

        try:
            self.rouge = Rouge()
        except Exception as e:
            logging.error(f"Failed to initialize Rouge scorer: {e}")
            raise

        # Validate word lists on init
        if not isinstance(self.config.positive_words, list):
            raise ValueError("Config positive_words must be a list.")
        if not isinstance(self.config.negative_words, list):
            raise ValueError("Config negative_words must be a list.")
        if not isinstance(self.config.critical_negative_words, list):
            raise ValueError("Config critical_negative_words must be a list.")

        self.chencherry = SmoothingFunction().method1  # Pre-init BLEU smoothing

        # Pre-process word lists for faster lookup? (e.g., lowercase, set)
        self._positive_words_set = set(
            w.lower() for w in self.config.positive_words if w
        )
        self._negative_words_set = set(
            w.lower() for w in self.config.negative_words if w
        )
        self._critical_negative_words_set = set(
            w.lower() for w in self.config.critical_negative_words if w
        )

        logging.debug("SimpleRewardFunction initialized.")

    def reset(self):
        """Resets episodic state (non-repetition history)."""
        self.previous_reasoning_steps = []
        if self.verbose:
            logging.debug("SimpleRewardFunction state reset for new episode.")

    def calculate_reward(
        self, generated_text: str, reference_text: str
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculates a comprehensive reward score, breaking down components for Thinking and Answer blocks.
        """
        # --- Initial Validation & Metrics Setup ---
        if not isinstance(generated_text, str) or not generated_text.strip():
            logging.warning(
                "calculate_reward received empty or invalid generated_text. Using min reward."
            )
            detailed_metrics = self._get_zeroed_metrics("Invalid generated_text")
            return self.config.min_reward_clip, detailed_metrics
        if not isinstance(reference_text, str):
            logging.warning(
                "calculate_reward received non-string reference_text. Treating as empty."
            )
            reference_text = ""

        detailed_metrics = self._initialize_metrics_dict()

        # --- Extract Content & Check Format ---
        think_start, think_end, answer_start, answer_end = self._get_tags()
        required_tags = [think_start, think_end, answer_start, answer_end]

        format_correct, format_issues = self._check_format(
            generated_text, required_tags
        )
        detailed_metrics["overall_metrics"]["format_score"] = (
            1.0 if format_correct else -1.0
        )  # Increased penalty
        if not format_correct:
            detailed_metrics["issues"].extend(format_issues)

        tag_presence_score = sum(0.25 for tag in required_tags if tag in generated_text)
        detailed_metrics["overall_metrics"]["tag_presence_score"] = tag_presence_score

        gen_think_content = self._extract_block_content(
            generated_text, think_start, think_end
        )
        gen_answer_content = self._extract_block_content(
            generated_text, answer_start, answer_end
        )
        ref_think_content = self._extract_block_content(
            reference_text, think_start, think_end
        )
        ref_answer_content = self._extract_block_content(
            reference_text, answer_start, answer_end
        )

        # --- Calculate Thinking Block Rewards ---
        if gen_think_content is not None:
            detailed_metrics["thinking_metrics"]["present"] = True
            detailed_metrics["thinking_metrics"]["content_raw"] = gen_think_content
            if gen_think_content:
                tm = detailed_metrics["thinking_metrics"]  # Alias for brevity
                tm["conciseness_score"] = self._calculate_conciseness_reward(
                    gen_think_content
                )
                tm["non_repetition_score"] = self._calculate_non_repetition_reward(
                    gen_think_content
                )
                current_positive_words = self._prepare_dynamic_positive_words(
                    ref_think_content
                )
                tm["positive_words_score"] = self._calculate_positive_words_reward(
                    gen_think_content, current_positive_words
                )
                (
                    tm["negative_words_factor"],
                    tm["critical_negative_words_factor"],
                ) = self._calculate_negative_words_penalty_factors(gen_think_content)
            else:
                detailed_metrics["issues"].append("Empty <thinking> block.")
                detailed_metrics["thinking_metrics"][
                    "conciseness_score"
                ] = -0.5  # Stronger penalty
        else:
            if format_correct:  # Format tags OK, but block missing? Problem.
                warn_msg = "Could not extract <thinking> block despite correct tags."
                logging.warning(warn_msg)
                detailed_metrics["issues"].append(warn_msg)
                detailed_metrics["overall_metrics"][
                    "format_score"
                ] -= 0.5  # Extra penalty

        # --- Calculate Answer Block Rewards ---
        if gen_answer_content is not None:
            detailed_metrics["answer_metrics"]["present"] = True
            detailed_metrics["answer_metrics"]["content_raw"] = gen_answer_content
            if gen_answer_content:
                if ref_answer_content:
                    try:
                        accuracy = self._calculate_accuracy_reward(
                            gen_answer_content, ref_answer_content
                        )
                        detailed_metrics["answer_metrics"]["accuracy_score"] = accuracy
                    except Exception as e:
                        logging.error(
                            f"Error in accuracy evaluation: {e}", exc_info=True
                        )
                        detailed_metrics["issues"].append(
                            f"Accuracy calculation failed: {e}"
                        )
                        detailed_metrics["answer_metrics"]["accuracy_score"] = 0.0
                else:
                    detailed_metrics["issues"].append(
                        "Reference answer missing, cannot calculate accuracy."
                    )
                    detailed_metrics["answer_metrics"]["accuracy_score"] = 0.0
            else:
                detailed_metrics["issues"].append("Empty <answer> block.")
                detailed_metrics["answer_metrics"][
                    "accuracy_score"
                ] = -0.5  # Stronger penalty
        else:
            if format_correct:
                warn_msg = "Could not extract <answer> block despite correct tags."
                logging.warning(warn_msg)
                detailed_metrics["issues"].append(warn_msg)
                detailed_metrics["overall_metrics"][
                    "format_score"
                ] -= 0.7  # Extra penalty

        # --- Combine Weighted Scores ---
        tm = detailed_metrics["thinking_metrics"]
        am = detailed_metrics["answer_metrics"]
        om = detailed_metrics["overall_metrics"]

        # Convert scores/factors to weighted impacts (penalty = weight * (score - 1) for factors/scores where 1 is good)
        non_rep_impact = self.config.think_non_repetition_weight * (
            tm["non_repetition_score"] - 1.0
        )
        neg_words_impact = self.config.think_negative_words_weight * (
            tm["negative_words_factor"] - 1.0
        )
        crit_neg_words_impact = self.config.think_critical_negative_words_weight * (
            tm["critical_negative_words_factor"] - 1.0
        )
        format_impact = self.config.reward_overall_format_weight * om["format_score"]
        tag_presence_impact = (
            self.config.reward_tag_presence_weight * om["tag_presence_score"]
        )
        conciseness_impact = (
            self.config.think_conciseness_weight * tm["conciseness_score"]
        )
        pos_words_impact = (
            self.config.think_positive_words_weight * tm["positive_words_score"]
        )
        accuracy_impact = self.config.answer_accuracy_weight * am["accuracy_score"]

        raw_total_reward = (
            format_impact
            + tag_presence_impact
            + non_rep_impact
            + conciseness_impact
            + pos_words_impact
            + neg_words_impact
            + crit_neg_words_impact
            + accuracy_impact  # Added critical impact
        )
        detailed_metrics["overall_reward"] = raw_total_reward

        # --- Normalize and Finalize ---
        normalized_reward = self._normalize_reward(raw_total_reward)
        detailed_metrics["normalized_reward"] = normalized_reward

        if self.verbose:
            log_msg = (
                f"Reward Calc: Format={om['format_score']:.1f}({format_impact:.2f}), "
                f"TagPres={om['tag_presence_score']:.2f}({tag_presence_impact:.2f}) | "
                f"THINK: NonRep={tm['non_repetition_score']:.2f}({non_rep_impact:.2f}), "
                f"Concise={tm['conciseness_score']:.2f}({conciseness_impact:.2f}), "
                f"PosW={tm['positive_words_score']:.2f}({pos_words_impact:.2f}), "
                f"NegWFactor={tm['negative_words_factor']:.2f}({neg_words_impact:.2f}), "
                f"CritNegWFactor={tm['critical_negative_words_factor']:.2f}({crit_neg_words_impact:.2f}) | "  # Added critical factor log
                f"ANSWER: Acc={am['accuracy_score']:.2f}({accuracy_impact:.2f}) |"
                f" --> RawTotal: {raw_total_reward:.3f}, Norm: {normalized_reward:.3f}"
            )
            logging.debug(log_msg)

        return normalized_reward, detailed_metrics

    # --- Helper/Calculation Functions ---

    def _get_tags(self) -> Tuple[str, str, str, str]:
        """Returns the configured tags."""
        return (
            self.config.special_tokens["think_start"],
            self.config.special_tokens["think_end"],
            self.config.special_tokens["answer_start"],
            self.config.special_tokens["answer_end"],
        )

    def _initialize_metrics_dict(self) -> Dict[str, Any]:
        """Returns a fresh dictionary for storing metrics."""
        return {
            "overall_reward": 0.0,
            "normalized_reward": 0.0,
            "issues": [],
            "overall_metrics": {"format_score": 0.0, "tag_presence_score": 0.0},
            "thinking_metrics": {
                "present": False,
                "content_raw": None,
                "conciseness_score": 0.0,
                "non_repetition_score": 1.0,
                "positive_words_score": 0.0,
                "negative_words_factor": 1.0,
                "critical_negative_words_factor": 1.0,  # Added critical factor
            },
            "answer_metrics": {
                "present": False,
                "content_raw": None,
                "accuracy_score": 0.0,
            },
        }

    def _get_zeroed_metrics(self, issue_message: str) -> Dict[str, Any]:
        """Returns a metrics dict representing a minimal/error state."""
        metrics = self._initialize_metrics_dict()
        metrics["issues"].append(issue_message)
        metrics["overall_metrics"]["format_score"] = -1.0  # Penalize format heavily
        # Calculate raw score based on this state for normalization baseline
        metrics["overall_reward"] = (
            self.config.reward_overall_format_weight
            * metrics["overall_metrics"]["format_score"]
        )
        metrics["normalized_reward"] = self._normalize_reward(metrics["overall_reward"])
        # Ensure clipping
        metrics["normalized_reward"] = np.clip(
            metrics["normalized_reward"],
            self.config.min_reward_clip,
            self.config.max_reward_clip,
        )
        return metrics

    def _extract_block_content(
        self, text: str, start_tag: str, end_tag: str
    ) -> Optional[str]:
        """Extracts content within the first occurrence of start_tag...end_tag block."""
        if not text or not isinstance(text, str):
            return None
        try:
            pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            return match.group(1).strip() if match else None
        except Exception as e:
            logging.error(
                f"Error extracting content [{start_tag}...{end_tag}]: {e}",
                exc_info=False,
            )  # Less noisy log
            return None

    def _check_format(
        self, text: str, required_tags: List[str]
    ) -> Tuple[bool, List[str]]:
        """Checks for presence, uniqueness, order, and separation of tags."""
        warnings = []
        tag_positions = {}
        missing, multiple = [], []

        for tag in required_tags:
            indices = [m.start() for m in re.finditer(re.escape(tag), text)]
            count = len(indices)
            if count == 0:
                missing.append(tag)
                tag_positions[tag] = -1
            elif count > 1:
                multiple.append(tag)
                tag_positions[tag] = indices[0]
            else:
                tag_positions[tag] = indices[0]

        if missing:
            warnings.append(f"Missing: {', '.join(missing)}")
        if multiple:
            warnings.append(f"Multiple: {', '.join(multiple)}")
        if missing or multiple:
            return False, warnings

        t0, t1, t2, t3 = required_tags
        p0, p1, p2, p3 = (
            tag_positions[t0],
            tag_positions[t1],
            tag_positions[t2],
            tag_positions[t3],
        )

        if not (p0 < p1 < p2 < p3):
            warnings.append(f"Order incorrect: TS={p0}, TE={p1}, AS={p2}, AE={p3}")
            return False, warnings

        separator = text[p1 + len(t1) : p2]
        if separator.strip():
            warnings.append(f"Non-whitespace between {t1} and {t2}")
            return False, warnings

        return True, []

    def _prepare_dynamic_positive_words(
        self, reference_think_content: Optional[str]
    ) -> Set[str]:
        """Creates the positive word set, potentially adding words from reference thinking."""
        # Start with the pre-processed set from config
        current_positive_words_set = set(self._positive_words_set)  # Make a copy

        if not reference_think_content or not isinstance(reference_think_content, str):
            return current_positive_words_set

        try:
            ref_words = reference_think_content.lower().split()
            # Consider punctuation stripping more robustly if needed
            filtered_ref_words = {
                word.strip(string.punctuation)
                for word in ref_words
                if word.strip(string.punctuation)
                # and word.lower() not in STOPWORDS
                and len(word.strip(string.punctuation)) > 2
                and word.lower()
                # not in self._negative_words_set  # Check against regular negative
                and word.lower()
                not in self._critical_negative_words_set  # Check against critical negative
            }

            if filtered_ref_words:
                # Add unique filtered words to the current set
                new_words_added = filtered_ref_words - current_positive_words_set
                if new_words_added:
                    current_positive_words_set.update(new_words_added)
                    if self.verbose:
                        logging.debug(
                            f"Added {len(new_words_added)} dynamic positive words from reference."
                        )
        except Exception as e_filter:
            logging.warning(
                f"Could not process reference thinking content for dynamic words: {e_filter}"
            )

        return current_positive_words_set

    def _calculate_accuracy_reward(
        self, predicted_answer: str, reference_answer: str
    ) -> float:
        """Calculates accuracy for the ANSWER BLOCK using cosine similarity, BLEU, and ROUGE-L."""
        if not predicted_answer or not reference_answer:
            return 0.0
        max_len = 5000
        pred = predicted_answer[:max_len].strip()
        ref = reference_answer[:max_len].strip()
        if not pred or not ref:
            return 0.0

        try:
            cos_sim, bleu_val, rouge_l_f1 = 0.0, 0.0, 0.0
            # Cosine Similarity
            try:
                vectorizer = CountVectorizer(
                    stop_words="english" if STOPWORDS else None
                ).fit([pred, ref])
                vecs = vectorizer.transform([pred, ref])
                if vecs.shape[1] > 0:
                    cos_sim = float(cosine_similarity(vecs)[0, 1])
            except Exception:
                pass  # Ignore vectorizer errors for robustness

            # BLEU Score
            try:
                pred_tokens, ref_tokens = pred.split(), [ref.split()]
                if pred_tokens and ref_tokens[0]:
                    bleu_val = bleu_score.sentence_bleu(
                        ref_tokens,
                        pred_tokens,
                        weights=(0.4, 0.3, 0.2, 0.1),
                        smoothing_function=self.chencherry,
                    )
            except Exception:
                pass  # Ignore BLEU errors

            # ROUGE-L Score
            try:
                rouge_scores = self.rouge.get_scores(pred, ref, avg=True)
                rouge_l_f1 = rouge_scores["rouge-l"]["f"] if rouge_scores else 0.0
            except Exception:
                pass  # Ignore ROUGE errors

            # Weighted combination
            reward = 0.4 * cos_sim + 0.4 * rouge_l_f1 + 0.2 * bleu_val
            return max(0.0, min(float(reward), 1.5))  # Clip 0 to 1.5

        except Exception as e:
            logging.error(
                f"Unexpected error in _calculate_accuracy_reward: {e}", exc_info=False
            )
            return 0.0

    def _calculate_non_repetition_reward(self, think_content: str) -> float:
        """Calculates a reward penalizing repetition within the THINKING content (score 0-1)."""
        if not think_content:
            return 1.0
        try:
            steps = [
                s.strip()
                for s in re.split(
                    r"[\.\?\!]\s+|\n\s*[-*]?\s*|\n\s*\d+\.\s+", think_content
                )
                if s.strip()
                and len(s.split()) >= self.config.non_repetition_min_step_words
            ]
            if not steps:
                return 1.0

            history = self.previous_reasoning_steps[
                -self.config.non_repetition_compare_depth :
            ]
            if not history:
                self.previous_reasoning_steps.extend(steps)
                self.previous_reasoning_steps = self.previous_reasoning_steps[
                    -self.config.non_repetition_history_length :
                ]
                return 1.0

            max_sim = [0.0] * len(steps)
            try:
                vecs = CountVectorizer().fit_transform(steps + history)
                if vecs.shape[1] > 0:
                    sim_matrix = cosine_similarity(
                        vecs[: len(steps)], vecs[len(steps) :]
                    )
                    if sim_matrix.size > 0:
                        max_sim = np.max(sim_matrix, axis=1).tolist()
            except Exception:
                pass  # Ignore vectorization/similarity errors

            reward = 1.0
            if max_sim:
                penalty = max(
                    0.0, np.mean(max_sim) - self.config.non_repetition_threshold
                )
                denominator = max(1e-8, 1.0 - self.config.non_repetition_threshold)
                reward = max(0.0, 1.0 - (penalty / denominator))

            self.previous_reasoning_steps.extend(steps)
            self.previous_reasoning_steps = self.previous_reasoning_steps[
                -self.config.non_repetition_history_length :
            ]
            return float(reward)
        except Exception as e:
            logging.error(f"Error in non-repetition: {e}", exc_info=False)
            return 0.0

    def _calculate_conciseness_reward(self, think_content: str) -> float:
        """Calculates reward based on THINKING word count relative to target (score -1 to 1)."""
        if not think_content:
            return 0.0
        num_words = len(think_content.split())
        if num_words == 0:
            return 0.0

        target = self.config.conciseness_target_tokens
        tolerance = self.config.conciseness_tolerance_ratio * target
        deviation = abs(num_words - target)

        if deviation <= tolerance:
            return 1.0
        else:
            excess = deviation - tolerance
            denom = max(1e-8, tolerance)
            penalty_units = excess / denom
            reward = 1.0 - penalty_units * 0.5  # Linear decrease
            return float(max(-1.0, reward))

    def _calculate_positive_words_reward(
        self, think_content: str, positive_words_set: Set[str]
    ) -> float:
        """Calculates reward based on positive words in THINKING content (score 0-1)."""
        if not think_content or not positive_words_set:
            return 0.0
        content_lower = " " + think_content.lower() + " "
        total_count = 0
        try:
            # Simple counting based on the dynamic set passed in
            # This method is approximate for multi-word phrases
            words_in_content = content_lower.split()
            # Count unique positive words found
            found_positive_words = {
                word for word in words_in_content if word in positive_words_set
            }
            total_count = len(
                found_positive_words
            )  # Reward variety? or total occurrences? Using variety here.

            # Alternative: Count total occurrences
            # total_count = sum(content_lower.count(f" {word} ") for word in positive_words_set)

        except Exception as e:
            logging.error(f"Error counting positive words: {e}", exc_info=False)
            return 0.0

        target = max(1e-8, self.config.max_positive_words_count)
        return float(min(total_count / target, 1.0))

    def _calculate_negative_words_penalty_factors(
        self, think_content: str
    ) -> Tuple[float, float]:
        """
        Calculates penalty factors (0-1, where 1=no penalty) for regular and critical negative words.
        Returns: (regular_negative_factor, critical_negative_factor)
        """
        if not think_content:
            return 1.0, 1.0

        content_lower = " " + think_content.lower() + " "
        regular_excess_count = 0
        critical_found_count = 0

        try:
            # Check regular negative words for excess count
            for word in self._negative_words_set:
                count = content_lower.count(f" {word} ")  # Approximate count
                regular_excess_count += max(
                    0, count - 1
                )  # Count beyond the first occurrence

            # Check critical negative words for presence
            for word in self._critical_negative_words_set:
                if f" {word} " in content_lower:
                    critical_found_count += 1  # Count occurrences of critical words

        except Exception as e:
            logging.error(f"Error counting negative words: {e}", exc_info=False)
            return 0.0, 0.0  # Max penalty on error

        # Calculate factor for regular negative words based on excess
        reg_denom = max(1e-8, self.config.max_negative_words_excess)
        regular_factor = max(0.0, 1.0 - (regular_excess_count / reg_denom))

        # Calculate factor for critical negative words based on presence
        # Simple model: penalty increases with each critical word found, max penalty after ~3 critical words?
        # Let's use a sharper penalty: reduce factor significantly even for one occurrence.
        # Factor = 1 / (1 + critical_found_count * penalty_multiplier) ?
        # Simpler: linear decrease, max penalty after 1 or 2 critical words.
        crit_penalty_per_word = 0.5  # Example: Max penalty after 2 critical words
        critical_factor = max(0.0, 1.0 - (critical_found_count * crit_penalty_per_word))

        return float(regular_factor), float(critical_factor)

    # def _normalize_reward(self, reward: float) -> float:
    #     """Normalizes the reward using EMA and standard deviation."""
    #     if not isinstance(reward, (float, int, np.number)) or not np.isfinite(reward):
    #         logging.warning(
    #             f"Invalid raw reward for normalization: {reward}. Using 0.0."
    #         )
    #         reward = 0.0

    #     alpha = self.config.reward_ema_alpha
    #     delta = reward - self.reward_ema
    #     self.reward_ema += alpha * delta
    #     variance_ema = (1.0 - alpha) * (self.reward_std**2) + alpha * (delta**2)
    #     self.reward_std = math.sqrt(max(variance_ema, 1e-8))

    #     if self.reward_std > 1e-6:
    #         normalized = (reward - self.reward_ema) / self.reward_std
    #     else:
    #         normalized = reward - self.reward_ema

    #     clipped = np.clip(
    #         normalized, self.config.min_reward_clip, self.config.max_reward_clip
    #     )

    #     if not np.isfinite(clipped):
    #         logging.warning(f"Non-finite normalized reward ({clipped}). Clamping.")
    #         clipped = np.nan_to_num(
    #             clipped,
    #             nan=0.0,
    #             posinf=self.config.max_reward_clip,
    #             neginf=self.config.min_reward_clip,
    #         )
    #     return float(clipped)


    def _normalize_reward(self, reward: float) -> float:
        """
        Normalizes the reward using EMA and standard deviation,
        after clipping the raw reward to configured bounds.
        """
        # 1) Clip the raw reward immediately
        try:
            raw = float(reward)
        except (ValueError, TypeError):
            logging.warning(f"Invalid raw reward for normalization: {reward}. Using 0.0.")
            raw = 0.0

        raw = np.clip(
            raw,
            self.config.min_reward_clip,
            self.config.max_reward_clip,
        )

        # 2) Update EMA and EMA‑based variance/std
        alpha = self.config.reward_ema_alpha
        delta = raw - self.reward_ema
        self.reward_ema += alpha * delta

        variance_ema = (
            (1.0 - alpha) * (self.reward_std ** 2)
            + alpha * (delta ** 2)
        )
        self.reward_std = math.sqrt(max(variance_ema, 1e-8))

        # 3) Z‑score the clipped raw reward
        if self.reward_std > 1e-6:
            normalized = (raw - self.reward_ema) / self.reward_std
        else:
            normalized = raw - self.reward_ema

        return float(normalized)

    def calculate_reference_reward(
        self, reference_text: str
    ) -> Tuple[float, Dict[str, Any]]:
        """Calculates the theoretical 'max' reward score for a given reference completion."""
        if not isinstance(reference_text, str) or not reference_text.strip():
            return self.config.min_reward_clip, self._get_zeroed_metrics(
                "Invalid reference text for max reward calculation"
            )

        original_history = deepcopy(self.previous_reasoning_steps)
        original_ema = self.reward_ema
        original_std = self.reward_std
        self.previous_reasoning_steps = []  # Reset history for ideal calc

        normalized_reward, detailed_metrics = self.calculate_reward(
            reference_text, reference_text
        )

        self.previous_reasoning_steps = original_history
        self.reward_ema = original_ema
        self.reward_std = original_std

        # Sanity check accuracy for ref vs ref
        if (
            detailed_metrics["answer_metrics"]["present"]
            and detailed_metrics["answer_metrics"]["accuracy_score"] < 0.95
        ):  # Expect near perfect score
            logging.debug(
                f"Reference reward calc accuracy low ({detailed_metrics['answer_metrics']['accuracy_score']:.3f})"
            )

        return normalized_reward, detailed_metrics

    def calculate_min_reward(self) -> Tuple[float, Dict[str, Any]]:
        """Calculates the reward for a minimally compliant placeholder text."""
        min_text = self.config.min_placeholder_reward_text

        original_history = deepcopy(self.previous_reasoning_steps)
        original_ema = self.reward_ema
        original_std = self.reward_std
        self.previous_reasoning_steps = []

        # Compare minimal text to itself
        normalized_reward, detailed_metrics = self.calculate_reward(min_text, min_text)

        self.previous_reasoning_steps = original_history
        self.reward_ema = original_ema
        self.reward_std = original_std

        return normalized_reward, detailed_metrics
