import re
import random
from typing import List, Dict, Optional, Tuple, Set, Any
from collections import Counter, defaultdict
import string
import math
import statistics  # Currently unused, but available if needed for stats problems

# from llama_rl.long_context import LongContext

REASONING_MODULES = [
    {
        "id": "1",
        ":module": "How could I devise an experiment to help solve that problem?. is it possible to represcent the problem in a markdown table for each prospective?",
    },
    {
        "id": "2",
        ":module": "Make a list of ideas for solving this problem, and apply them one by one to see if progress can be made.. Can you use symbolic and autometa based reasoning to solve this?",
    },
    {
        "id": "3",
        ":module": "How could I measure progress on this problem? How about you simulate a real world scenior? thinking thru each role  turn by turn in a markdown table monolouge",
    },
    {
        "id": "4",
        ":module": "How can I simplify the problem so that it is easier to solve?.",
    },
    {"id": "5", ":module": "What are the key assumptions underlying this problem?"},
    {
        "id": "6",
        ":module": "What are the potential risks and drawbacks of each solution?. Ask many followup questions in for clarity, try answering those in your <thinking> if unanswered, asked user. Did you track the contraints?",
    },
    {
        "id": "7",
        ":module": "What are the alternative perspectives on this problem? Can we verify your current progress as a critic?",
    },
    {
        "id": "8",
        ":module": "What are the long-term implications of this problem and its solutions?. How  about we switch roles so you can think more about it ?",
    },
    {
        "id": "9",
        ":module": "How can I break down this problem into smaller, manageable parts?. What is the biggest challenge?",
    },
    {
        "id": "10",
        ":module": "Critical Thinking: Analyze the problem from different perspectives.. Can you work backwards?",
    },
    {
        "id": "11",
        ":module": "Try creative thinking to generate innovative solutions. Is it a one way door or two way door ?",
    },
    {
        "id": "12",
        ":module": "Seek input and collaboration from others.Do you have all the metrics needed for informed decision making?",
    },
    {
        "id": "13",
        ":module": "Use systems thinking: Consider the problem as part of a larger system.. What leadership principles would help sovle this problem?",
    },
    {
        "id": "14",
        ":module": "Use risk analysis: Evaluate potential risks and tradeoffs..",
    },
    {
        "id": "15",
        ":module": "Use reflective thinking: Step back and reflect on your approach.. Can you backtrack and criticise your work as a second person?",
    },
    {
        "id": "16",
        ":module": "What is the core issue that needs to be addressed? How could you define suceess metrics for each stage?",
    },
    {
        "id": "17",
        ":module": "What are the underlying causes of the problem? How would you score your answer?",
    },
    {
        "id": "18",
        ":module": "Have any solutions been tried before? What were the outcomes?.",
    },
    {"id": "19", ":module": "What obstacles might arise in solving this problem?"},
    {"id": "20", ":module": "Are there any relevant data or information available?"},
    {
        "id": "21",
        ":module": "Who is directly affected by the problem? What are their perspectives?.",
    },
    {
        "id": "22",
        ":module": "What resources are needed to tackle the problem effectively?.",
    },
    {"id": "23", ":module": "How can progress or success be measured?"},
    {"id": "24", ":module": "What indicators or metrics can be used?"},
    {"id": "25", ":module": "Is the problem technical or conceptual?"},
    {"id": "26", ":module": "Does the problem involve a physical constraint?"},
    {"id": "27", ":module": "Is the problem related to human behavior or psychology?"},
    {
        "id": "28",
        ":module": "Does the problem involve decision-making under uncertainty?.",
    },
    {
        "id": "29",
        ":module": "Is the problem analytical and requires data analysis or modeling?.",
    },
    {
        "id": "30",
        ":module": "Is the problem a design challenge that requires creative solutions?.",
    },
    {"id": "31", ":module": "Does the problem require addressing systemic issues?"},
    {"id": "32", ":module": "Is the problem time-sensitive or urgent?"},
    {"id": "33", ":module": "What kinds of solutions typically work for this problem?"},
    {"id": "34", ":module": "Have a guess about other possible solutions."},
    {
        "id": "35",
        ":module": "Imagine the current best solution is wrong; what are other ways to thinking about the problem?.",
    },
    {
        "id": "36",
        ":module": "How can you modify the current best solution? Can your work be improved?",
    },
    {
        "id": "37",
        ":module": "Create an entirely new solution, ignoring the current best..",
    },
    {
        "id": "38",
        ":module": "Let’s thinking step by step, using markdown table with steps , sub steps, and STAR cols.",
    },
    {"id": "39", ":module": "Make a step-by-step plan and explain it well."},
    # --- Phase 1: Problem Definition & Understanding ---
    {
        "id": "PD-1",
        ":module": "Define the core problem: State the central issue concisely. What is the desired outcome or goal state? What are the explicit non-goals?",
    },
    {
        "id": "PD-2",
        ":module": "Identify Key Assumptions: List the underlying beliefs or conditions taken for granted in the problem statement or context. How critical is each assumption? What happens if an assumption is wrong?",
    },
    {
        "id": "PD-3",
        ":module": "Clarify Scope and Context: What are the boundaries of this problem? What external factors or systems influence it? Is there relevant history or background?",
    },
    {
        "id": "PD-4",
        ":module": "Identify Stakeholders: Who is affected by this problem, directly or indirectly? List their perspectives, needs, and potential concerns. Consider presenting this in a markdown table.",
    },
    {
        "id": "PD-5",
        ":module": "Define Success Metrics: How will we know if the problem is successfully solved or progress is made? List specific, measurable indicators (KPIs or metrics) for evaluation.",
    },
    {
        "id": "PD-6",
        ":module": "Identify Constraints & Resources: What limitations exist (time, budget, technical, personnel, ethical, legal)? What resources (data, tools, expertise) are available or required?",
    },
    # --- Phase 2: Analysis & Information Gathering ---
    {
        "id": "AN-1",
        ":module": "Root Cause Analysis: Explore the underlying causes, not just symptoms. Use techniques like the '5 Whys' or a Fishbone (Ishikawa) diagram structure in your thinking. What is the fundamental driver?",
    },
    {
        "id": "AN-2",
        ":module": "Gather Relevant Data: What information or data is needed to understand the problem better or evaluate solutions? Where can this data be found? Outline a plan to gather it if necessary.",
    },
    {
        "id": "AN-3",
        ":module": "Analyze Previous Attempts: Have solutions been tried before (internally or elsewhere)? Describe them and their outcomes (successes, failures, lessons learned).",
    },
    {
        "id": "AN-4",
        ":module": "Break Down the Problem: Decompose the main problem into smaller, more manageable sub-problems or components. Identify the dependencies between these parts.",
    },
    {
        "id": "AN-5",
        ":module": "Problem Categorization: Is this primarily a technical, conceptual, logistical, behavioral, design, analytical, or systemic problem? Does it involve uncertainty or physical constraints?",
    },
    # --- Phase 3: Idea Generation & Solution Design ---
    {
        "id": "IG-1",
        ":module": "Brainstorm Potential Solutions: Generate a diverse list of ideas without initial judgment. Consider unconventional approaches. Use creative thinking techniques.",
    },
    {
        "id": "IG-2",
        ":module": "Develop Experimental Approaches: How could an experiment be designed to test a potential solution or key assumption? Define the hypothesis, method, data collection, and success criteria for the experiment.",
    },
    {
        "id": "IG-3",
        ":module": "Simplify the Problem: Can the problem be restated in a simpler form? Solve the simpler version first to gain insights. What is the Minimum Viable Solution?",
    },
    {
        "id": "IG-4",
        ":module": "Use Analogical Reasoning: Think of similar problems in different domains. What solutions were used there? How could those solutions be adapted to this problem?",
    },
    {
        "id": "IG-5",
        ":module": "Consider Symbolic/Automata Reasoning: Can the problem be modeled using symbols, state machines, or formal logic? Apply these models to explore potential solution paths or dead ends.",
    },
    {
        "id": "IG-6",
        ":module": "Challenge the Current Best Idea: Assume the leading solution is flawed or insufficient. Generate alternative solutions by fundamentally rethinking the approach.",
    },
    # --- Phase 4: Evaluation & Selection ---
    {
        "id": "EV-1",
        ":module": "Evaluate Solutions Against Criteria: Assess each proposed solution against the defined success metrics and constraints. Use a matrix or table for comparison if helpful.",
    },
    {
        "id": "EV-2",
        ":module": "Risk Analysis: For each promising solution, identify potential risks, drawbacks, and unintended consequences. Assess the likelihood and impact of each risk. Are mitigation strategies possible?",
    },
    {
        "id": "EV-3",
        ":module": "Consider Long-Term Implications: What are the future effects of implementing each solution? Think about scalability, maintainability, and adaptability.",
    },
    {
        "id": "EV-4",
        ":module": "Ethical Considerations: Analyze potential ethical issues related to the problem or its solutions. Are there fairness, bias, privacy, or societal impact concerns?",
    },
    {
        "id": "EV-5",
        ":module": "Decision Type: Is this decision a 'one-way door' (hard to reverse) or a 'two-way door' (easy to reverse)? How does this influence the required level of certainty or the choice of solution?",
    },
    # --- Phase 5: Planning & Implementation ---
    {
        "id": "PL-1",
        ":module": "Develop a Step-by-Step Plan: Outline the sequence of actions needed to implement the chosen solution. Define milestones, responsibilities, and timelines. Use a STAR (Situation, Task, Action, Result) format for key steps if applicable.",
    },
    {
        "id": "PL-2",
        ":module": "Identify Implementation Obstacles: What practical challenges might arise during implementation (e.g., resistance to change, technical hurdles, resource gaps)? How can these be proactively addressed?",
    },
    {
        "id": "PL-3",
        ":module": "Resource Allocation: Detail the specific resources (personnel, budget, tools) needed for implementation and how they will be allocated.",
    },
    {
        "id": "PL-4",
        ":module": "Communication Plan: How will the solution and implementation plan be communicated to relevant stakeholders? What information needs to be shared, when, and by whom?",
    },
    # --- Phase 6: Reflection & Refinement ---
    {
        "id": "RR-1",
        ":module": "Simulate and Role-Play: Mentally walk through the solution's implementation or use. Consider different user roles or scenarios. Use a markdown table to track turns or perspectives in a monologue format.",
    },
    {
        "id": "RR-2",
        ":module": "Critical Self-Review: Step back and act as a critic of your own reasoning and proposed solution. Where are the weak points? What was missed? Could the analysis be stronger?",
    },
    {
        "id": "RR-3",
        ":module": "Work Backwards: Start from the desired outcome and trace the steps back to the present. Does this reveal any gaps or necessary intermediate steps in the plan?",
    },
    {
        "id": "RR-4",
        ":module": "Reflect on the Process: Review the problem-solving approach used. What worked well? What could be improved next time? Capture key learnings.",
    },
    {
        "id": "RR-5",
        ":module": "Seek External Feedback: Identify points where input or collaboration from others (experts, users, team members) would be valuable. What specific questions should be asked?",
    },
    # --- Meta-Cognitive & Systemic Approaches ---
    {
        "id": "MC-1",
        ":module": "Systems Thinking: View the problem as part of a larger interconnected system. Map out system components and relationships. How do changes in one part affect others? Identify feedback loops.",
    },
    {
        "id": "MC-2",
        ":module": "Apply Leadership Principles: Consider relevant leadership principles (e.g., ownership, bias for action, dive deep, invent and simplify). How would applying these principles shape the approach or solution?",
    },
    {
        "id": "MC-3",
        ":module": "Identify Necessary Clarifications: Review the current understanding. What specific questions need to be answered to proceed effectively? If information is missing, state what it is and why it's needed."
        # This replaces the repetitive "Ask many followup questions..." pattern.
    },
    {
        "id": "MC-4",
        ":module": "Synthesize and Summarize Progress: Briefly summarize the current state of understanding, key findings, proposed next steps, and any outstanding questions or challenges.",
    },
    # --- Phase 1: Problem Definition & Understanding ---
    {
        "id": "PD-1",
        ":module": "Define the core problem: State the central issue concisely. What is the desired outcome or goal state? What are the explicit non-goals?",
    },
    {
        "id": "PD-2",
        ":module": "Identify Key Assumptions: List the underlying beliefs or conditions taken for granted. How critical is each assumption? What happens if an assumption is wrong? What evidence supports/contradicts each?",
    },
    {
        "id": "PD-3",
        ":module": "Clarify Scope and Context: Define strict boundaries. Detail relevant historical context, precedents, and influencing external systems (e.g., economic, social, technological trends).",
    },
    {
        "id": "PD-4",
        ":module": "Identify Stakeholders & Perspectives: Map out all affected parties. Detail their explicit and implicit needs, values, potential conflicts, and power dynamics. Use a table for clarity.",
    },
    {
        "id": "PD-5",
        ":module": "Define Success Metrics (Multi-level): Define primary success metrics, secondary indicators, and leading/lagging indicators. How will unintended consequences be monitored?",
    },
    {
        "id": "PD-6",
        ":module": "Identify Constraints & Resources (Dynamic): List hard constraints (non-negotiable) and soft constraints (negotiable). Map available resources and identify critical resource gaps. How might constraints change over time?",
    },
    {
        "id": "PD-7",
        ":module": "Problem Framing & Reframing: Articulate the current frame used to view the problem. Generate 3-5 alternative frames (e.g., frame as an opportunity, a system failure, a communication issue). How does reframing change the perceived problem?",
    },
    {
        "id": "PD-8",
        ":module": "Trace Problem Evolution: Detail the history of this problem. When did it emerge? What attempts were made to address it previously? How has its nature changed over time?",
    },
    {
        "id": "PD-9",
        ":module": "Define the 'Ideal' vs. 'Acceptable' Solution: Describe the characteristics of a truly transformative, ideal solution, even if seemingly unattainable. Contrast this with the minimum acceptable solution criteria.",
    },
    {
        "id": "PD-10",
        ":module": "Distill the Problem Essence: What is the absolute core, underlying tension or principle at the heart of this problem? Express it as a fundamental question or paradox.",
    },
    # --- Phase 2: Analysis & Information Gathering ---
    {
        "id": "AN-1",
        ":module": "Root Cause Analysis (Systemic): Use '5 Whys' deeply. Map causes using a Fishbone diagram. Identify feedback loops (reinforcing and balancing) that perpetuate the problem.",
    },
    {
        "id": "AN-2",
        ":module": "Information Inventory & Gap Analysis: List all known relevant information. Explicitly identify critical knowledge gaps. Assess the reliability and potential biases of existing data. Outline a prioritised information gathering plan.",
    },
    {
        "id": "AN-3",
        ":module": "Analyze Previous Attempts (Deep Dive): For prior solutions, analyze *why* they succeeded or failed, considering context, execution, assumptions, and overlooked factors. What specific, transferable lessons were learned?",
    },
    {
        "id": "AN-4",
        ":module": "Decomposition & Dependency Mapping: Break the problem into sub-components. Map the dependencies (sequential, parallel, interdependent) between them. Which components are most critical or leveraged?",
    },
    {
        "id": "AN-5",
        ":module": "Problem Categorization (Multi-dimensional): Classify the problem across multiple axes (e.g., technical vs. adaptive, simple vs. complex vs. chaotic, data-rich vs. data-poor). How does this classification guide the approach?",
    },
    {
        "id": "AN-6",
        ":module": "Second & Third Order Consequence Analysis: For the primary *causes* of the problem, trace the ripple effects. What are the immediate consequences (1st order), the consequences of those consequences (2nd order), and subsequent effects (3rd order)?",
    },
    {
        "id": "AN-7",
        ":module": "Interacting Systems Mapping: Identify the key systems involved (e.g., technical, social, economic, political). Map how they interact, influence each other, and contribute to the current state. Where are points of leverage or friction?",
    },
    {
        "id": "AN-8",
        ":module": "Identify Paradoxes and Contradictions: Are there inherent tensions or contradictory requirements within the problem space (e.g., speed vs. accuracy, cost vs. quality, standardization vs. customization)? Articulate these clearly.",
    },
    {
        "id": "AN-9",
        ":module": "Quantify Uncertainty: Where knowledge is incomplete, attempt to quantify the uncertainty. Use ranges, probabilities, or confidence levels where possible. How sensitive is the problem understanding to these uncertainties?",
    },
    {
        "id": "AN-10",
        ":module": "Cognitive Bias Audit (Self-Correction): Review the analysis so far. Actively look for potential cognitive biases (e.g., confirmation bias, anchoring, availability heuristic, sunk cost fallacy). How might these biases be distorting the understanding? Propose steps to mitigate them.",
    },
    # --- Phase 3: Idea Generation & Solution Design ---
    {
        "id": "IG-1",
        ":module": "Divergent Brainstorming (Unconstrained): Generate a wide array of ideas, including impractical or radical ones. Use techniques like SCAMPER (Substitute, Combine, Adapt, Modify, Put to another use, Eliminate, Reverse).",
    },
    {
        "id": "IG-2",
        ":module": "Develop Experimental Probes: Design small, low-cost experiments ('probes') not just to test solutions, but to learn more about the problem space or test critical assumptions under real conditions.",
    },
    {
        "id": "IG-3",
        ":module": "Strategic Simplification & MVP Definition: Identify the absolute core function a solution must perform (Minimum Viable Product). How can the problem be simplified to address this core first, allowing iterative expansion?",
    },
    {
        "id": "IG-4",
        ":module": "Cross-Domain Analogical Reasoning: Systematically search for analogous problems/solutions in conceptually distant fields (e.g., biology, military strategy, art, urban planning). Detail the analogy and adapt the foreign solution principle.",
    },
    {
        "id": "IG-5",
        ":module": "Formal Modeling & Simulation: If applicable, build a formal model (mathematical, logical, agent-based) of the problem or proposed solution. Simulate its behavior under different conditions to gain insights.",
    },
    {
        "id": "IG-6",
        ":module": "Assumption Reversal & Inversion: Take key assumptions identified earlier and reverse them. What new solution possibilities emerge if the opposite were true? Invert the problem: how could one *cause* this problem?",
    },
    {
        "id": "IG-7",
        ":module": "Extreme Brainstorming (Constraint Breaking): Generate ideas that violate existing constraints (cost, time, physics). Analyze *why* they are impossible. Can the principle behind the 'impossible' idea be achieved differently?",
    },
    {
        "id": "IG-8",
        ":module": "Principle-Based Solution Design: Instead of concrete solutions, define the core principles a successful solution *must* embody (e.g., 'must be decentralized', 'must enhance user agency', 'must be self-healing'). Generate solutions that fit these principles.",
    },
    {
        "id": "IG-9",
        ":module": "Future Backcasting: Imagine a detailed, ideal future state (e.g., 10 years out) where the problem is fully resolved. Work backward from that future to identify the necessary breakthroughs, decisions, and steps required to reach it.",
    },
    {
        "id": "IG-10",
        ":module": "Solution Ecosystem Design: Instead of a single solution, design a portfolio or ecosystem of complementary solutions that address different facets of the problem or cater to different stakeholders.",
    },
    # --- Phase 4: Evaluation & Selection ---
    {
        "id": "EV-1",
        ":module": "Multi-Criteria Decision Analysis (MCDA): Evaluate top solutions against weighted criteria (success metrics, constraints, risks, principles). Use a structured scoring method (e.g., weighted sum, AHP).",
    },
    {
        "id": "EV-2",
        ":module": "Systemic Risk Analysis & Cascading Failures: For each solution, map potential direct risks AND how these risks might cascade through interconnected systems. Identify potential systemic vulnerabilities.",
    },
    {
        "id": "EV-3",
        ":module": "Long-Term Viability & Adaptability Assessment: Evaluate solutions based on projected future trends, scalability, maintainability, and adaptability to changing conditions. How easily can the solution evolve?",
    },
    {
        "id": "EV-4",
        ":module": "Ethical Matrix Analysis: Analyze solutions against core ethical principles (e.g., autonomy, justice, beneficence, non-maleficence) and stakeholder impacts. Identify and plan mitigation for ethical trade-offs.",
    },
    {
        "id": "EV-5",
        ":module": "Decision Type & Reversibility Cost: Re-assess if it's a one-way or two-way door. Quantify the potential cost (financial, reputational, strategic) of reversing the decision if it proves wrong.",
    },
    {
        "id": "EV-6",
        ":module": "Pre-Mortem Analysis (Detailed Scenarios): Imagine the chosen solution has failed 1 year post-implementation. Brainstorm all plausible reasons for failure, categorizing them (e.g., technical, market, adoption, execution). Develop specific mitigation/contingency plans for the top 3-5 failure modes.",
    },
    {
        "id": "EV-7",
        ":module": "Sensitivity & Robustness Analysis: Identify key variables/assumptions underpinning the solution's success. Analyze how changes in these variables impact the expected outcome. How robust is the solution to variance?",
    },
    {
        "id": "EV-8",
        ":module": "Value Conflict Resolution Strategy: Explicitly identify conflicts between desirable values (e.g., efficiency vs equity). Define a strategy or framework for how these conflicts will be navigated or balanced during implementation.",
    },
    {
        "id": "EV-9",
        ":module": "Resilience & Antifragility Test: How would the solution perform under extreme stress or unexpected shocks (e.g., sudden market crash, supply chain disruption, regulatory change)? Does it merely survive (resilience) or potentially benefit (antifragility)?",
    },
    {
        "id": "EV-10",
        ":module": "Comparative Advantage Check: Is our organization/team uniquely positioned to implement this specific solution effectively compared to alternatives or compared to other actors?",
    },
    # --- Phase 5: Planning & Implementation ---
    {
        "id": "PL-1",
        ":module": "Phased Rollout & Milestones: Develop a detailed, phased implementation plan with clear milestones, deliverables, Go/No-Go decision points, and feedback loops for each phase.",
    },
    {
        "id": "PL-2",
        ":module": "Obstacle & Contingency Planning (Proactive): For obstacles identified in EV-6/PL-2, develop specific, actionable contingency plans (Plan B, Plan C) including triggers for activation.",
    },
    {
        "id": "PL-3",
        ":module": "Resource & Dependency Scheduling: Create a detailed resource allocation plan and timeline, mapping critical path dependencies. Identify potential bottlenecks and mitigation strategies.",
    },
    {
        "id": "PL-4",
        ":module": "Stakeholder Communication & Alignment Plan: Develop a tailored communication strategy for different stakeholder groups, focusing on building buy-in, managing expectations, and gathering feedback throughout implementation.",
    },
    {
        "id": "PL-5",
        ":module": "Monitoring & Evaluation Framework: Define the specific data to be collected, frequency, methods, and responsibilities for monitoring progress against metrics (from PD-5) and detecting unintended consequences.",
    },
    {
        "id": "PL-6",
        ":module": "Adaptive Management Plan: Design the implementation process to be flexible. Define specific points and criteria for reviewing progress and adapting the plan based on new information or changing circumstances.",
    },
    {
        "id": "PL-7",
        ":module": "Knowledge Management & Transfer Strategy: Plan how key insights, rationale, learnings, and documentation from the problem-solving and implementation process will be captured, shared, and preserved.",
    },
    {
        "id": "PL-8",
        ":module": "Pilot Program Design: If applicable, design a small-scale pilot implementation. Define its scope, objectives, success criteria, duration, and learning goals before full rollout.",
    },
    # --- Phase 6: Reflection & Refinement ---
    {
        "id": "RR-1",
        ":module": "Multi-Perspective Simulation: Simulate the implemented solution from the viewpoint of different key stakeholders (users, operators, affected third parties). Use role-playing or written narratives to uncover usability issues, conflicts, or unforeseen impacts.",
    },
    {
        "id": "RR-2",
        ":module": "Structured Self-Critique (Red Teaming): Appoint a 'red team' perspective (even if just mentally) whose sole purpose is to rigorously challenge the chosen solution, the implementation plan, and the underlying reasoning. Document the strongest criticisms.",
    },
    {
        "id": "RR-3",
        ":module": "Work Backwards from Failure/Success: Trace back from potential failure points (identified in Pre-Mortem) to identify weak links in the plan. Also, trace back from the ideal success state to ensure all enabling steps are included.",
    },
    {
        "id": "RR-4",
        ":module": "Process & Learning Debrief: After key phases or completion, conduct a structured debrief. What worked well in the reasoning/planning process? What were the major challenges? What specific improvements can be made for future problems?",
    },
    {
        "id": "RR-5",
        ":module": "External Feedback Integration Plan: Define how external feedback (from users, experts, pilot programs) will be systematically collected, analyzed, and integrated into refining the solution or implementation.",
    },
    {
        "id": "RR-6",
        ":module": "Epistemological Review: Reflect on the nature and limits of the knowledge used. How confident are we in the evidence and conclusions? What level of uncertainty remains? What fundamental assumptions *could* still be wrong?",
    },
    {
        "id": "RR-7",
        ":module": "Alternative History Exploration: Consider 1-2 key decision points during the process. If a different path had been chosen, sketch out how the reasoning and outcome might have diverged. What does this reveal about the chosen path?",
    },
    {
        "id": "RR-8",
        ":module": "Simplify & Communicate Core Insight: After the complex analysis, can the core problem insight, the solution's logic, and the implementation strategy be distilled into a simple, clear, and compelling narrative?",
    },
    # --- Meta-Cognitive & Systemic Approaches ---
    {
        "id": "MC-1",
        ":module": "Dynamic Systems Thinking & Modeling: Map the system with feedback loops, delays, stocks, and flows. Use causal loop diagrams or stock-and-flow diagrams. Identify high-leverage intervention points within the system dynamics.",
    },
    {
        "id": "MC-2",
        ":module": "Apply Diverse Mental Models: Consciously apply different mental models (e.g., Pareto Principle, Occam's Razor, Network Effects, Critical Mass, Equilibrium) to analyze the problem and potential solutions. Which models yield the most insight?",
    },
    {
        "id": "MC-3",
        ":module": "Explicit Reasoning Audit Trail: Document the chain of thought step-by-step: Key question -> Information used -> Assumption made -> Inference drawn -> Confidence level. Identify any significant logical leaps or gaps.",
    },
    {
        "id": "MC-4",
        ":module": "Synthesize & Structure Knowledge: Organize the accumulated insights, analyses, and plans into a coherent structure (e.g., using a logic tree, concept map, or structured report outline). Ensure consistency and clarity.",
    },
    {
        "id": "MC-5",
        ":module": "Information Flow & Processing Audit: Reflect on how information was gathered, filtered, prioritized, and synthesized during the reasoning process. Identify potential bottlenecks, biases, or inefficiencies in this meta-process.",
    },
    {
        "id": "MC-6",
        ":module": "Level of Abstraction Agility: Practice deliberately shifting perspective between granular details, operational processes, strategic goals, and high-level principles/purpose. How does the problem/solution look different at each level? Ensure alignment across levels.",
    },
    {
        "id": "MC-7",
        ":module": "Paradigm Shift Inquiry: Could this problem be fundamentally redefined or dissolved by adopting a radically different worldview, set of values, or technological paradigm? Explore at least one such alternative paradigm.",
    },
    {
        "id": "MC-8",
        ":module": "Check for Underlying Principles: What fundamental scientific, mathematical, psychological, or economic principles underpin the problem and potential solutions? Are solutions aligned with or fighting against these principles?",
    },
]


# --- Supporting Class for Anti-Forgetting (Expanded Library) ---
# NOTE: MathProblemLibrary class definition is omitted here for brevity,
# assume it's the same as provided in the original prompt.
class MathProblemLibrary:
    """
    Expanded library of math problems for the anti-forgetting mechanism. (v9 Update)
    Generates problems across different types and difficulties.
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize the math problem library."""
        self.config = config if config else {}

    def get_problem(
        self, domain: str = "arithmetic", difficulty: str = "medium"
    ) -> Tuple[Optional[str], Optional[str]]:
        """Get a math problem and solution of specified type/difficulty."""
        generators = {
            "arithmetic": self.generate_arithmetic_problem,
            "algebra": self.generate_algebra_problem,
            "geometry": self.generate_geometry_problem,
            "probability": self.generate_probability_problem,
        }
        generator_func = generators.get(domain, self.generate_arithmetic_problem)
        return generator_func(difficulty)

    def _safe_eval(self, expression: str) -> Optional[float]:
        """Safely evaluate simple mathematical expressions."""
        try:
            allowed_names = {
                k: v for k, v in math.__dict__.items() if not k.startswith("__")
            }
            allowed_names.update({"abs": abs, "round": round})
            # Basic check against disallowed characters - still not fully secure
            if re.search(
                r"[^0-9.+\-*/^() \t]|import|eval|exec|lambda|__|class|def", expression
            ):
                # print(f"Warning: Potentially unsafe eval characters blocked: {expression}")
                # return None # Block potentially unsafe eval
                pass  # Allow simple functions like pow, sqrt for now
            code = compile(expression, "<string>", "eval")
            # Evaluate with limited builtins and allowed math functions
            result = eval(code, {"__builtins__": {}}, allowed_names)
            return float(result)
        except Exception as e:
            # print(f"Error evaluating expression '{expression}': {e}")
            return None

    def generate_arithmetic_problem(
        self, difficulty: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """Generate arithmetic problems with increasing complexity."""
        # (Implementation from v8 - slightly more variation)
        try:
            if difficulty == "easy":
                a, b = random.randint(1, 25), random.randint(1, 25)
                op_sym = random.choice(["+", "-"])
                if op_sym == "-" and b > a:
                    a, b = b, a
                problem = f"Calculate: {a} {op_sym} {b}"
                answer_val = self._safe_eval(f"{a}{op_sym}{b}")
            elif difficulty == "medium":
                a, b, c = (
                    random.randint(1, 50),
                    random.randint(1, 25),
                    random.randint(2, 10),
                )
                ops = random.sample(["+", "-", "*", "/"], 2)
                # Ensure division or multiplication is included
                if "*" not in ops and "/" not in ops:
                    ops[random.randint(0, 1)] = random.choice(["*", "/"])
                # Construct carefully to avoid division by zero if '/' is second op
                if ops[1] == "/":
                    c = random.randint(1, 10)  # Ensure c is not 0 for division
                elif ops[0] == "/" and b == 0:
                    b = random.randint(1, 25)  # Ensure b is not 0

                problem = f"Calculate: {a} {ops[0]} {b} {ops[1]} {c}"
                # Handle potential division by zero before eval
                expr = f"{a}{ops[0]}{b}{ops[1]}{c}"
                if f"/{0}" in expr.replace(" ", ""):
                    return None, None  # Basic check
                answer_val = self._safe_eval(expr)
            else:  # hard
                a, b = random.randint(10, 100), random.randint(5, 50)
                c = random.randint(2, 15)
                d = random.randint(1, 30)
                ops = random.sample(["+", "-", "*", "/"], 3)
                # Construct carefully with parentheses
                expr = f"(({a}{ops[0]}{b}){ops[1]}{c}){ops[2]}{d}"
                # Avoid division by zero
                if "/0" in expr.replace(" ", ""):
                    return None, None
                problem = f"Calculate: (({a} {ops[0]} {b}) {ops[1]} {c}) {ops[2]} {d}"
                answer_val = self._safe_eval(expr)

            if answer_val is None:
                return None, None
            answer = (
                str(int(answer_val))
                if answer_val == int(answer_val)
                else f"{answer_val:.2f}"
            )
            return problem, answer
        except Exception as e:
            print(f"Error generating arithmetic problem: {e}")
            return None, None

    def generate_algebra_problem(
        self, difficulty: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """Generate simple linear algebra problems with integer solutions."""
        # (Implementation from v8)
        try:
            target_x = random.randint(-15, 15)
            if difficulty == "easy":  # ax + b = c
                a = random.choice([i for i in range(-7, 8) if i != 0])
                b = random.randint(-20, 20)
                c = a * target_x + b
                x = target_x
                problem = (
                    f"Solve for x: {a}x + {b} = {c}"
                    if b >= 0
                    else f"Solve for x: {a}x - {abs(b)} = {c}"
                )
                answer = f"x = {int(x)}"
            else:  # medium/hard: ax + b = cx + d
                a, c = random.randint(-15, 15), random.randint(-15, 15)
                while a == c or a == 0 or c == 0:
                    a, c = random.randint(-15, 15), random.randint(-15, 15)
                b = random.randint(-30, 30)
                d = (a - c) * target_x + b
                x = target_x
                a_str = f"{a}x" if abs(a) != 1 else ("x" if a == 1 else "-x")
                c_str = f"{c}x" if abs(c) != 1 else ("x" if c == 1 else "-x")
                b_str = f"+ {b}" if b > 0 else f"- {abs(b)}" if b < 0 else ""
                d_str = f"+ {d}" if d > 0 else f"- {abs(d)}" if d < 0 else ""
                problem = f"Solve for x: {a_str} {b_str} = {c_str} {d_str}"
                answer = f"x = {int(x)}"
            return problem, answer
        except Exception as e:
            print(f"Error generating algebra problem: {e}")
            return None, None

    def generate_geometry_problem(
        self, difficulty: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """Generate basic geometry problems."""
        # (Expanded slightly from v8)
        try:
            shape = random.choice(["rectangle", "circle", "triangle", "square"])
            if shape == "rectangle":
                l, w = random.randint(2, 25), random.randint(2, 25)
                calc = random.choice(["area", "perimeter"])
                if calc == "area":
                    problem = f"Calculate the area of a rectangle with length {l} and width {w}."
                    answer = str(l * w)
                else:
                    problem = f"Calculate the perimeter of a rectangle with length {l} and width {w}."
                    answer = str(2 * (l + w))
            elif shape == "square":
                s = random.randint(2, 20)
                calc = random.choice(["area", "perimeter"])
                if calc == "area":
                    problem = f"Calculate the area of a square with side length {s}."
                    answer = str(s**2)
                else:
                    problem = (
                        f"Calculate the perimeter of a square with side length {s}."
                    )
                    answer = str(4 * s)
            elif shape == "circle":
                r = random.randint(2, 20)
                calc = random.choice(["area", "circumference"])
                pi = 3.14159
                if calc == "area":
                    problem = f"Calculate the area of a circle with radius {r}. (Use pi ≈ 3.14)"
                    answer = f"{round(pi * r**2, 2)}"
                else:
                    problem = f"Calculate the circumference of a circle with radius {r}. (Use pi ≈ 3.14)"
                    answer = f"{round(2 * pi * r, 2)}"
            else:  # triangle (right-angled)
                a, b = random.randint(3, 25), random.randint(3, 25)
                calc = random.choice(["area", "hypotenuse"])
                if calc == "area":
                    problem = f"Calculate the area of a right-angled triangle with legs of length {a} and {b}."
                    answer = str(0.5 * a * b)
                else:
                    problem = f"Find the hypotenuse of a right-angled triangle with legs of length {a} and {b}."
                    answer = f"{round(math.sqrt(a**2 + b**2), 2)}"
            return problem, answer
        except Exception as e:
            print(f"Error generating geometry problem: {e}")
            return None, None

    def generate_probability_problem(
        self, difficulty: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """Generate basic probability problems."""
        # (Expanded slightly from v8)
        try:
            event_type = random.choice(["coin", "dice", "cards"])
            if event_type == "coin":
                n_flips = (
                    random.randint(1, 2)
                    if difficulty == "easy"
                    else random.randint(2, 5)
                )
                target = "heads" if random.random() < 0.5 else "tails"
                if n_flips == 1:
                    problem = f"What is the probability of flipping a fair coin once and getting {target}?"
                    answer = "0.5"
                else:
                    problem = f"What is the probability of flipping a fair coin {n_flips} times and getting {target} every time?"
                    answer = f"{round(0.5**n_flips, 4)}"
            elif event_type == "dice":
                n_rolls = 1 if difficulty == "easy" else 2
                if n_rolls == 1:
                    target_num = random.randint(1, 6)
                    problem = f"What is the probability of rolling a fair six-sided die once and getting a {target_num}?"
                    answer = f"{round(1/6, 4)}"
                else:  # 2 rolls
                    op = random.choice(["sum", "specific"])
                    if op == "sum":
                        target_sum = random.randint(3, 11)
                        count = sum(
                            1
                            for i in range(1, 7)
                            for j in range(1, 7)
                            if i + j == target_sum
                        )
                        prob = count / 36
                        problem = f"What is the probability of rolling two fair six-sided dice and getting a sum of {target_sum}?"
                        answer = f"{round(prob, 4)}"
                    else:
                        t1, t2 = random.randint(1, 6), random.randint(1, 6)
                        problem = f"What is the probability of rolling two fair six-sided dice and getting a {t1} on the first roll and a {t2} on the second roll?"
                        answer = f"{round(1/36, 4)}"
            else:  # cards (simple)
                card_type = random.choice(["suit", "rank"])
                if card_type == "suit":
                    suit = random.choice(["hearts", "diamonds", "clubs", "spades"])
                    problem = f"What is the probability of drawing a {suit} from a standard 52-card deck?"
                    answer = f"{round(13/52, 4)}"  # 0.25
                else:  # rank
                    rank = random.choice(
                        [
                            "Ace",
                            "King",
                            "Queen",
                            "Jack",
                            "10",
                            "9",
                            "8",
                            "7",
                            "6",
                            "5",
                            "4",
                            "3",
                            "2",
                        ]
                    )
                    problem = f"What is the probability of drawing a {rank} from a standard 52-card deck?"
                    answer = f"{round(4/52, 4)}"
            return problem, answer
        except Exception as e:
            print(f"Error generating probability problem: {e}")
            return None, None


# --- Main Guidance Generator Class (MODIFIED) ---


class COTGuidanceGenerator:
    """
    Main class for generating guidance prompts to improve CoT reasoning,
    with integrated adaptive loss calculation, topic modeling proxy,
    math handling, anti-forgetting support, strategic guidance,
    and **optional enforcement of <thinking>/<answer> tags**. (v10 - Tag Enforcement)
    """

    # Constants and Defaults
    STAGE_INITIATION = "INITIATION"
    STAGE_DECOMPOSITION = "DECOMPOSITION"
    STAGE_REASONING = "REASONING"
    STAGE_VERIFICATION = "VERIFICATION"
    STAGE_UNKNOWN = "UNKNOWN"
    DEFAULT_STOP_WORDS = set(
        [
            "a",
            "an",
            "the",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "as",
            "is",
            "am",
            "are",
            "was",
            "were",
            "be",
            "being",
            "been",
            "it",
            "its",
            "i",
            "you",
            "he",
            "she",
            "we",
            "they",
            "me",
            "him",
            "her",
            "us",
            "them",
            "my",
            "your",
            "his",
            "its",
            "our",
            "their",
            "and",
            "or",
            "but",
            "so",
            "if",
            "then",
            "what",
            "which",
            "who",
            "whom",
            "this",
            "that",
            "these",
            "those",
            "how",
            "why",
            "when",
            "where",
            "let's",
            "okay",
            "step",
            "solve",
            "calculate",
            "find",
            "explain",
            "can",
            "could",
            "will",
            "would",
            "should",
            "about",
            "also",
            "just",
            "get",
            "go",
            "make",
            "know",
            "see",
            "use",
            "well",
            "very",
        ]
    )
    DEFAULT_CONFIG = {
        # ... (Existing config entries from v9) ...
        "cot_initiation_keywords": [
            "let's think step by step",
            "step 1",
            "first,",
            "break it down",
            "plan:",
            "approach:",
            "initial thoughts:",
        ],
        "uncertainty_keywords": [
            "not sure",
            "don't know",
            "maybe",
            "perhaps",
            "error",
            "failed",
            "confused",
            "difficult",
            "unsure",
            "might be",
        ],
        "final_answer_markers": [
            "final answer:",
            "therefore, the answer is",
            "so the answer is",
            "the result is",
            "conclusion:",
            "in summary:",
            "the final value is",
        ],
        "repetition_ngram_size": 8,
        "repetition_threshold": 2,
        "min_part_length_words": 4,
        "min_keyword_length": 4,
        "stop_words": DEFAULT_STOP_WORDS,
        "domain_keywords": {
            "math": [
                "calculate",
                "solve",
                "equation",
                "variable",
                "proof",
                "theorem",
                "sum",
                "product",
                "integral",
                "fraction",
                "decimal",
                "percent",
                "interest",
                "principal",
                "rate",
                "geometry",
                "algebra",
                "arithmetic",
                "unit",
                "conversion",
                "formula",
                "graph",
                "plot",
            ],
            "logic": [
                "deduce",
                "infer",
                "premise",
                "conclusion",
                "consistent",
                "valid",
                "argument",
                "syllogism",
                "fallacy",
                "logical",
                "if",
                "then",
                "not",
                "and",
                "or",
                "implies",
            ],
            "finance": [
                "interest",
                "principal",
                "rate",
                "loan",
                "investment",
                "compounded",
                "annuity",
                "present value",
                "future value",
                "return",
                "stock",
                "bond",
                "yield",
                "portfolio",
                "asset",
                "liability",
                "budget",
            ],
            "statistics": [
                "mean",
                "median",
                "mode",
                "average",
                "standard deviation",
                "variance",
                "sample",
                "population",
                "probability",
                "distribution",
                "correlation",
                "regression",
                "hypothesis test",
                "p-value",
                "confidence interval",
                "data",
            ],
        },
        "math_guidance_templates": {
            "arithmetic": [
                "Could you write out each calculation step separately?",
                "What operation should be performed first according to order of operations (PEMDAS/BODMAS)?",
                "Try working with one operation at a time, showing your work.",
                "Double-check your arithmetic calculations.",
            ],
            "algebra": [
                "Can you isolate the variable on one side of the equation?",
                "What happens if you perform the same operation on both sides of the equation?",
                "Try substituting known values to simplify the expression.",
                "Check your algebraic manipulation steps carefully.",
                "Define the variables clearly first.",
            ],
            "geometry": [
                "What formula applies to this shape or calculation (e.g., area, volume, perimeter)?",
                "Can you break this shape into simpler components?",
                "Try drawing a diagram and labeling the given values and what you need to find.",
                "Are you using the correct units?",
            ],
            "probability": [
                "What's the total possible number of outcomes (the sample space)?",
                "What are the favorable outcomes for this event?",
                "Are these events independent or dependent? How does that affect the calculation?",
                "Try expressing this using probability notation (P(A), P(B|A), etc.)",
            ],
            "unit_conversion": [
                "What is the conversion factor between the units?",
                "Should you multiply or divide by the conversion factor?",
                "Set up the conversion using dimensional analysis (canceling units).",
                "Ensure the final answer has the correct units.",
            ],
            "finance": [
                "Start by identifying the key financial variables (principal, interest rate, time period, payments).",
                "Apply the appropriate financial formula (e.g., simple interest, compound interest, annuity).",
                "Make sure you're using the correct time units (years, months) consistent with the interest rate.",
                "Consider if interest is compounded annually, semi-annually, monthly, etc.",
            ],
            "statistics": [
                "Identify whether you're working with sample data or population data.",
                "What statistical measure is required (e.g., mean, median, mode, standard deviation, variance)?",
                "Choose the appropriate formula for the required measure.",
                "Consider the significance of outliers if present in the data.",
                "What does this statistical measure tell you about the data?",
            ],
            "general_math": [
                "Focus on identifying the key numbers and the operations required.",
                "Break the problem down into smaller mathematical steps.",
                "Ensure you're following the correct order of operations.",
            ],
        },
        "anti_forgetting": {
            "enabled": False,
            "math_frequency": 0.1,
            "difficulty_distribution": {"easy": 0.4, "medium": 0.5, "hard": 0.1},
            "domains": ["arithmetic", "algebra", "geometry", "probability"],
        },
        "loss_parameters": {
            "structure_weight": 1.0,
            "content_weight": 1.5,
            "math_weight": 1.5,
            "topic_weight": 0.5,
            "guidance_weight": 0.5,
        },
        "topic_modeling": {"num_topics": 1, "num_words": 10, "tfidf_min_df": 1},
        "reasoning_patterns": {
            "logic": [
                r"(?i)\bif\b.*\bthen\b",
                r"(?i)premise.*conclusion",
                r"(?i)(?:assume|suppose).*contradiction",
            ],
            "math": [
                r"(?i)formula.*=",
                r"(?i)substitut.*\=",
                r"(?i)\bstep\b.*\=",
                r"(?i)let\s+\w+\s*=",
                r"(?i)solve for",
            ],
            "general": [
                r"(?i)pros.*cons",
                r"(?i)compare.*contrast",
                r"(?i)cause.*effect",
            ],
        },
        "adaptive_memory": {"epsilon": 0.1},  # Epsilon for Epsilon-Greedy
        # --- NEW: Tag Enforcement Configuration ---
        "enforce_thinking_answer_tags": False,  # Default is OFF
        "thinking_tag_open": "<thinking>",
        "thinking_tag_close": "</thinking>",
        "answer_tag_open": "<answer>",
        "answer_tag_close": "</answer>",
        "tag_structure_penalty": 0.8,  # Penalty multiplier in structure loss if required tags are missing
        "tag_content_use_answer": True,  # If True and tags enforced, content loss primarily compares <answer> content
    }

    def __init__(self, total_steps: int, config: Optional[Dict] = None):
        """Initialize the generator with configuration."""
        self.total_steps = total_steps
        # Deep merge config (simplified version)
        temp_config = self.DEFAULT_CONFIG.copy()
        if config:
            for key, value in config.items():
                if isinstance(value, dict) and isinstance(temp_config.get(key), dict):
                    # Ensure nested dicts like loss_parameters are updated correctly
                    temp_config[key].update(value)
                else:
                    temp_config[key] = value
        self.config = temp_config

        # --- Load configuration (including new tag settings) ---
        self.cot_initiation_keywords = set(self.config["cot_initiation_keywords"])
        self.uncertainty_keywords = set(self.config["uncertainty_keywords"])
        self.final_answer_markers = set(self.config["final_answer_markers"])
        self.step_pattern = re.compile(
            r"(?i)(\bstep\s*\d+\b|\b\d+\.\s|\bfirst(?:ly)?\b|\bsecond(?:ly)?\b|\bthird(?:ly)?\b|\bnext\b|->)"
        )
        self.sentence_split_pattern = re.compile(r"(?<=[.?!])\s+")
        self.repetition_ngram_size = self.config["repetition_ngram_size"]
        self.repetition_threshold = self.config["repetition_threshold"]
        self.min_part_length_words = self.config["min_part_length_words"]
        self.min_keyword_length = self.config["min_keyword_length"]
        self.stop_words = self.config["stop_words"]
        self.domain_keywords = self.config["domain_keywords"]
        self.math_guidance_templates = self.config["math_guidance_templates"]
        self.loss_weights = self.config["loss_parameters"]
        self.topic_config = self.config["topic_modeling"]
        self.reasoning_patterns = {
            domain: [re.compile(p) for p in patterns]
            for domain, patterns in self.config.get("reasoning_patterns", {}).items()
        }
        self.adaptive_memory_config = self.config["adaptive_memory"]
        # New tag config
        self.enforce_tags = self.config.get("enforce_thinking_answer_tags", False)
        self.thinking_open = self.config.get("thinking_tag_open", "<thinking>")
        self.thinking_close = self.config.get("thinking_tag_close", "</thinking>")
        self.answer_open = self.config.get("answer_tag_open", "<answer>")
        self.answer_close = self.config.get("answer_tag_close", "</answer>")
        self.tag_penalty_factor = self.config.get("tag_structure_penalty", 0.8)
        self.tag_content_use_answer = self.config.get("tag_content_use_answer", True)

        # Compile tag regex patterns (case-insensitive, DOTALL)
        self.thinking_pattern = re.compile(
            re.escape(self.thinking_open) + r"(.*?)" + re.escape(self.thinking_close),
            re.DOTALL | re.IGNORECASE,
        )
        self.answer_pattern = re.compile(
            re.escape(self.answer_open) + r"(.*?)" + re.escape(self.answer_close),
            re.DOTALL | re.IGNORECASE,
        )

        # --- Internal state tracking ---
        self._last_guidance_type = None
        # Store effectiveness keyed by context tuple (domain, stage, struggle_type) -> {scores: [], avg: 0.5}
        self.guidance_effectiveness = defaultdict(lambda: {"scores": [], "avg": 0.5})
        self._last_gen_reasoning_length = 0
        self._last_total_loss = None  # Store previous loss for effectiveness calc
        self.loss_history = []

        # --- Anti-Forgetting Setup ---
        self.anti_forgetting_config = self.config["anti_forgetting"]
        if self.anti_forgetting_config.get("enabled", False):
            self.math_problem_library = MathProblemLibrary(
                config=self.config.get("math_problem_library_config")
            )
        else:
            self.math_problem_library = None

    # --- Core Analysis Methods (MODIFIED/NEW) ---

    def _extract_tagged_content(self, text: str) -> Dict[str, Optional[str]]:
        """Extracts content within the first <thinking> and <answer> tags."""
        thinking_match = self.thinking_pattern.search(text)
        answer_match = self.answer_pattern.search(text)
        return {
            "thinking_content": thinking_match.group(1).strip()
            if thinking_match
            else None,
            "answer_content": answer_match.group(1).strip() if answer_match else None,
        }

    def _extract_final_answer(
        self, text: str, analysis: Optional[Dict] = None
    ) -> Optional[str]:
        """
        Extract final answer. Prioritizes <answer> tag content if enforcement is on
        and the tag is present in the analysis results.
        """
        # If tag enforcement is on and analysis provides an answer_content, use it.
        if self.enforce_tags and analysis and analysis.get("has_answer_tag"):
            return analysis.get("answer_content")

        # Fallback to original marker-based extraction if enforcement is off
        # or if enforcement is on but the tag wasn't found (signaling a potential structure issue).
        text_lower = text.lower()
        extracted_answer = None
        latest_marker_pos = -1
        for marker in self.final_answer_markers:
            marker_lower = marker.lower()
            try:
                marker_pos = text_lower.rindex(marker_lower)
                if marker_pos > latest_marker_pos:
                    # Ensure we don't grab content from inside a <thinking> block if tags exist
                    preceding_text = text_lower[:marker_pos]
                    thinking_tag_index = preceding_text.rfind(
                        self.thinking_open.lower()
                    )
                    closing_thinking_tag_index = preceding_text.rfind(
                        self.thinking_close.lower()
                    )

                    # Only consider the marker if it's *not* inside a thinking block
                    # (or if thinking tags aren't present)
                    is_inside_thinking = thinking_tag_index != -1 and (
                        closing_thinking_tag_index == -1
                        or closing_thinking_tag_index < thinking_tag_index
                    )

                    if not is_inside_thinking:
                        potential_answer = text[marker_pos + len(marker) :].strip()
                        # Prevent grabbing huge chunks, limit to a few lines
                        lines_in_potential = potential_answer.splitlines()
                        if len(lines_in_potential) <= 5:  # Increased slightly
                            # Avoid grabbing the start of a new section if tags are missing
                            if (
                                self.thinking_open.lower()
                                not in lines_in_potential[0].lower()
                                and self.answer_open.lower()
                                not in lines_in_potential[0].lower()
                            ):
                                extracted_answer = potential_answer
                                latest_marker_pos = marker_pos

            except ValueError:
                continue

        if extracted_answer is not None:
            return extracted_answer

        # Fallback: Take the last line if it's short and doesn't look like code/tags
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if lines:
            last_line = lines[-1]
            if (
                len(last_line) < 150
                and len(last_line.split()) < 30
                and not last_line.lower().startswith(self.thinking_open.lower())
                and not last_line.lower().startswith(self.answer_open.lower())
                and not last_line.lower().endswith(self.thinking_close.lower())
                and not last_line.lower().endswith(self.answer_close.lower())
            ):
                return last_line

        # If tag enforcement is on and we couldn't find the tag OR a marker, return None
        # to indicate failure to extract the answer cleanly.
        if self.enforce_tags and analysis and not analysis.get("has_answer_tag"):
            return None

        return None  # Default if nothing found

    def _analyze_cot_structure(self, text: str, domain: str = "general") -> Dict:
        """
        Analyze CoT structure, markers, reasoning depth, basic domain patterns,
        and presence/content of <thinking>/<answer> tags.
        """
        text_lower = text.lower()
        tagged_content = self._extract_tagged_content(text)

        analysis = {
            "text": text,  # Store original text
            "has_thinking_tag": tagged_content["thinking_content"] is not None,
            "has_answer_tag": tagged_content["answer_content"] is not None,
            "thinking_content": tagged_content["thinking_content"],
            "answer_content": tagged_content["answer_content"],
            "has_initiation": any(
                keyword in text_lower for keyword in self.cot_initiation_keywords
            ),
            "step_count": len(self.step_pattern.findall(text)),
            "has_uncertainty": any(
                keyword in text_lower for keyword in self.uncertainty_keywords
            ),
            "length_chars": len(text),
            "reasoning_length_chars": 0,  # Will be updated
            "has_final_answer_marker": any(
                marker.lower() in text_lower for marker in self.final_answer_markers
            ),
            "detected_domain_pattern": False,
        }

        reasoning_part = text  # Default
        # If tags are present, use thinking content as the primary reasoning part
        if analysis["has_thinking_tag"]:
            reasoning_part = analysis["thinking_content"]
            analysis["reasoning_length_chars"] = len(reasoning_part)
        else:
            # If no thinking tag, fall back to extracting based on final answer markers (original logic)
            final_answer_via_marker = self._extract_final_answer(
                text
            )  # Use original extraction for this part
            if final_answer_via_marker and final_answer_via_marker in text:
                try:
                    last_answer_index = text.rindex(final_answer_via_marker)
                    potential_reasoning = text[:last_answer_index].strip()
                    # If answer tag exists but thinking doesn't, avoid including answer tag in reasoning
                    if (
                        analysis["has_answer_tag"]
                        and self.answer_open in potential_reasoning
                    ):
                        answer_tag_start = potential_reasoning.rfind(self.answer_open)
                        if answer_tag_start != -1:
                            potential_reasoning = potential_reasoning[
                                :answer_tag_start
                            ].strip()

                    reasoning_part = potential_reasoning
                    analysis["reasoning_length_chars"] = len(reasoning_part)
                except ValueError:
                    # If marker extraction failed to find index, use full text length (minus answer if possible)
                    analysis["reasoning_length_chars"] = len(text) - (
                        len(analysis["answer_content"] or "")
                        if analysis["has_answer_tag"]
                        else 0
                    )
            else:
                analysis["reasoning_length_chars"] = len(text) - (
                    len(analysis["answer_content"] or "")
                    if analysis["has_answer_tag"]
                    else 0
                )

        analysis["has_minimal_reasoning"] = analysis["reasoning_length_chars"] > 50

        # Domain pattern check (on reasoning part if possible, else full text)
        check_text_for_pattern = (
            reasoning_part if analysis["has_thinking_tag"] else text
        )
        domain_patterns = self.reasoning_patterns.get(domain, [])
        for pattern in domain_patterns:
            if pattern.search(check_text_for_pattern):
                analysis["detected_domain_pattern"] = True
                break

        return analysis

    def _detect_repetition(self, text: str) -> bool:
        """Detect excessive repetition in text."""
        # (Implementation from v8 - unmodified)
        min_words_for_check = self.repetition_ngram_size * (
            self.repetition_threshold + 1
        )
        words = re.findall(r"\b\w+\b", text.lower())
        if not text or len(words) < min_words_for_check:
            return False
        ngrams = [
            " ".join(words[i : i + self.repetition_ngram_size])
            for i in range(len(words) - self.repetition_ngram_size + 1)
        ]
        if not ngrams:
            return False
        ngram_counts = Counter(ngrams)
        return any(count > self.repetition_threshold for count in ngram_counts.values())

    def _breakdown_target(self, target_text: str) -> Optional[List[str]]:
        """Break down target text into logical parts."""
        # (Implementation from v8 - unmodified)
        parts = []
        step_matches = list(self.step_pattern.finditer(target_text))
        if len(step_matches) > 1:
            start_index = 0
            for i, match in enumerate(step_matches):
                part_end_index = match.start()
                part = target_text[start_index:part_end_index].strip()
                if len(part.split()) >= self.min_part_length_words:
                    parts.append(part)
                start_index = match.start()
            last_part = target_text[start_index:].strip()
            if len(last_part.split()) >= self.min_part_length_words:
                parts.append(last_part)
        if not parts or len(parts) < 2:
            sentences = self.sentence_split_pattern.split(target_text)
            parts = [
                s.strip()
                for s in sentences
                if len(s.split()) >= self.min_part_length_words
            ]
        return parts if parts and len(parts) > 1 else None

    def _infer_reasoning_stage(
        self, gen_analysis: Dict, conversation_history: List[Dict]
    ) -> str:
        """Infer current reasoning stage, considering tags if enforced."""
        # (Implementation from v8 - slightly modified logic based on tags)
        # Use has_answer_tag if enforcement is on, otherwise marker
        has_final_answer = (
            gen_analysis["has_answer_tag"]
            if self.enforce_tags
            else gen_analysis["has_final_answer_marker"]
        )

        if (
            not gen_analysis["has_minimal_reasoning"]
            and gen_analysis["step_count"] == 0
        ):
            assistant_turns = [
                msg for msg in conversation_history if msg.get("role") == "assistant"
            ]
            if len(assistant_turns) <= 1:
                return self.STAGE_INITIATION
            else:
                return (
                    self.STAGE_UNKNOWN
                )  # Unclear state if minimal reasoning fails later

        # If thinking tag is present but reasoning is short, likely decomposition
        if (
            gen_analysis["has_thinking_tag"]
            and gen_analysis["reasoning_length_chars"] < 150
            and gen_analysis["step_count"] < 2
        ):
            return self.STAGE_DECOMPOSITION
        # If initiation marker present but reasoning is short (and no thinking tag yet)
        if (
            gen_analysis["has_initiation"]
            and not gen_analysis["has_thinking_tag"]
            and gen_analysis["step_count"] < 2
            and gen_analysis["reasoning_length_chars"] < 150
        ):
            return self.STAGE_DECOMPOSITION

        # If minimal reasoning exists and no final answer identified
        if gen_analysis["has_minimal_reasoning"] and not has_final_answer:
            return self.STAGE_REASONING

        # If final answer identified OR reasoning is substantial
        if has_final_answer or (
            gen_analysis["has_minimal_reasoning"]
            and gen_analysis["reasoning_length_chars"] > 300
        ):
            return self.STAGE_VERIFICATION

        return self.STAGE_UNKNOWN

    # --- TF-IDF & Keyword Methods (Unmodified from v9) ---
    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        return re.findall(r"\b\w+\b", text)

    def _calculate_tf(self, tokens: List[str]) -> Counter:
        return Counter(tokens)

    def _calculate_idf(self, documents: List[List[str]]) -> Dict[str, float]:
        num_documents = len(documents)
        if num_documents == 0:
            return {}
        df = Counter()
        for doc_tokens in documents:
            df.update(set(doc_tokens))
        idf = {
            term: math.log(num_documents / (count + 1)) + 1
            for term, count in df.items()
        }
        return idf

    def _calculate_tfidf_vector_manual(
        self, text: str, idf: Dict[str, float], domain: str
    ) -> Dict[str, float]:
        tokens = self._tokenize(text)
        tf = self._calculate_tf(tokens)
        num_tokens = len(tokens)
        if num_tokens == 0:
            return {}
        vector = {
            term: (count / num_tokens) * idf.get(term, 0)
            for term, count in tf.items()
            if term in idf
        }
        norm = math.sqrt(sum(score**2 for score in vector.values()))
        if norm > 0:
            return {term: score / norm for term, score in vector.items()}
        return {}

    def _cosine_similarity(
        self, vec1: Dict[str, float], vec2: Dict[str, float]
    ) -> float:
        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum(vec1[x] * vec2[x] for x in intersection)
        sum1 = sum(vec1[x] ** 2 for x in vec1)
        sum2 = sum(vec2[x] ** 2 for x in vec2)
        denominator = math.sqrt(sum1) * math.sqrt(sum2)
        if not denominator:
            return 1.0 if not numerator else 0.0
        similarity = numerator / denominator
        return max(0.0, min(1.0, similarity))

    def _extract_keywords(
        self, text: str, domain: str = "general", num_keywords: Optional[int] = None
    ) -> Set[str]:
        text_clean = text.translate(str.maketrans("", "", string.punctuation))
        words = re.findall(r"\b\w+\b", text_clean.lower())
        domain_specific_keywords = set(self.domain_keywords.get(domain, []))
        current_stop_words = self.stop_words - domain_specific_keywords
        keywords = {
            word
            for word in words
            if len(word) >= self.min_keyword_length and word not in current_stop_words
        }
        if num_keywords:
            word_counts = Counter(w for w in words if w in keywords)
            return set(dict(word_counts.most_common(num_keywords)).keys())
        return keywords

    def _check_keyword_gap(
        self, gen_text: str, target_text: str, domain: str
    ) -> List[str]:
        gen_keywords = self._extract_keywords(gen_text, domain)
        target_keywords = self._extract_keywords(target_text, domain)
        missing_keywords = list(target_keywords - gen_keywords)
        return missing_keywords[:3]

    # --- Math Specific Methods (Unmodified from v9) ---
    def _detect_math_content(self, text: str) -> Dict[str, Any]:
        math_analysis = {
            "is_math_problem": False,
            "numbers_detected": [],
            "operations_detected": [],
            "math_type": None,
            "complexity": "low",
        }
        number_pattern = r"[-+]?\b\d+(?:\.\d+)?(?:e[-+]?\d+)?\b"
        math_analysis["numbers_detected"] = [
            float(n) for n in re.findall(number_pattern, text)
        ]
        operations = {
            "addition": ["+", "plus", "sum", "add", "total", "increase by"],
            "subtraction": [
                "-",
                "minus",
                "subtract",
                "difference",
                "less",
                "decrease by",
            ],
            "multiplication": [
                "*",
                "×",
                "times",
                "multiply",
                "product",
                "multiplied by",
            ],
            "division": [
                "/",
                "÷",
                "divide",
                "quotient",
                "per",
                "divided by",
                "ratio of",
            ],
            "exponentiation": [
                "^",
                "**",
                "squared",
                "cubed",
                "power",
                "exponent",
                "raised to the",
            ],
        }
        text_lower = text.lower()
        [
            math_analysis["operations_detected"].append(op)
            for op, m in operations.items()
            if any(mk in text_lower for mk in m)
        ]
        math_keywords = [
            "solve",
            "calculate",
            "find",
            "what is",
            "how many",
            "equation",
            "problem",
            "value",
            "compute",
            "evaluate",
            "math",
            "algebra",
            "geometry",
            "probability",
            "statistics",
        ]
        equation_pattern = (
            r"\b(?:\w|\d)+(?:\.\d+)?\s*(?:[+\-*/^=]|\*\*)\s*(?:\w|\d)+(?:\.\d+)?\b"
        )
        has_equation_pattern = bool(re.search(equation_pattern, text))
        if math_analysis["numbers_detected"] or has_equation_pattern:
            if (
                math_analysis["operations_detected"]
                or any(kw in text_lower for kw in math_keywords)
                or has_equation_pattern
            ):
                math_analysis["is_math_problem"] = True
        if math_analysis["is_math_problem"]:
            if any(
                x in text_lower
                for x in [
                    "interest",
                    "principal",
                    "rate",
                    "loan",
                    "investment",
                    "compounded",
                    "finance",
                ]
            ):
                math_analysis["math_type"] = "finance"
            elif any(
                x in text_lower
                for x in [
                    "mean",
                    "median",
                    "mode",
                    "average",
                    "standard deviation",
                    "variance",
                    "statistics",
                    "data set",
                    "sample",
                    "population",
                ]
            ):
                math_analysis["math_type"] = "statistics"
            elif any(
                x in text_lower
                for x in [
                    "probability",
                    "chance",
                    "likelihood",
                    "odds",
                    "event",
                    "dice",
                    "coin",
                ]
            ):
                math_analysis["math_type"] = "probability"
            elif any(
                unit in text_lower
                for unit in [
                    "meter",
                    "meters",
                    "m ",
                    "km",
                    "mile",
                    "miles",
                    "feet",
                    "foot",
                    "ft",
                    "inch",
                    "inches",
                    "gram",
                    "grams",
                    "kg",
                    "liter",
                    "liters",
                    "gallon",
                    "litre",
                    "litres",
                ]
            ) and any(
                x in text_lower
                for x in ["convert", "conversion", "how many", "in terms of"]
            ):
                math_analysis["math_type"] = "unit_conversion"
            elif any(
                x in text_lower
                for x in [
                    "area",
                    "volume",
                    "perimeter",
                    "triangle",
                    "circle",
                    "rectangle",
                    "angle",
                    "geometry",
                    "shape",
                    "degrees",
                ]
            ):
                math_analysis["math_type"] = "geometry"
            elif "=" in text or any(
                v in text_lower
                for v in ["solve for", "variable", " x ", " y ", " z ", " n "]
            ):
                math_analysis["math_type"] = "algebra"
            elif math_analysis["operations_detected"]:
                math_analysis["math_type"] = "arithmetic"
        num_ops = len(math_analysis["operations_detected"])
        num_nums = len(math_analysis["numbers_detected"])
        if (
            num_ops > 2
            or num_nums > 4
            or math_analysis["math_type"]
            in ["algebra", "geometry", "statistics", "finance"]
        ):
            math_analysis["complexity"] = "medium"
        if (
            num_ops > 3
            or num_nums > 6
            or "exponentiation" in math_analysis["operations_detected"]
        ):
            math_analysis["complexity"] = "high"
        return math_analysis

    def _decompose_math_problem(
        self, problem_text: str, math_analysis: Dict[str, Any]
    ) -> List[str]:
        steps = []
        if not math_analysis["is_math_problem"]:
            return steps
        steps.extend(
            [
                "Carefully read the problem and identify what needs to be found or calculated.",
                "Identify all the given numbers, units, and information.",
            ]
        )
        math_type = math_analysis.get("math_type")
        if math_type == "arithmetic":
            steps.extend(
                [
                    "Determine the sequence of operations needed (consider PEMDAS/BODMAS).",
                    "Perform each calculation step-by-step, showing intermediate results.",
                    "Check if the final answer makes sense (e.g., estimation).",
                ]
            )
        elif math_type == "algebra":
            steps.extend(
                [
                    "Define variables for any unknown quantities.",
                    "Translate the problem statement into one or more equations.",
                    "Solve the equation(s) using appropriate algebraic techniques.",
                    "Check the solution by substituting it back into the original equation(s).",
                ]
            )
        elif math_type == "geometry":
            steps.extend(
                [
                    "Identify the geometric shapes involved.",
                    "Recall or look up relevant formulas.",
                    "Label a diagram with the given information if helpful.",
                    "Apply the formulas and perform calculations.",
                    "Ensure the answer has the correct units.",
                ]
            )
        elif math_type == "probability":
            steps.extend(
                [
                    "Identify the total number of possible outcomes (sample space).",
                    "Identify the number of favorable outcomes.",
                    "Determine if events are independent or dependent.",
                    "Apply the appropriate probability formula.",
                ]
            )
        elif math_type == "unit_conversion":
            steps.extend(
                [
                    "Identify the starting unit and the target unit.",
                    "Find the correct conversion factor(s) between the units.",
                    "Set up the calculation to cancel the starting units and end with the target units (dimensional analysis).",
                    "Perform the multiplication/division.",
                ]
            )
        elif math_type == "finance":
            steps.extend(
                [
                    "Identify the financial variables provided (Principal, Rate, Time, Payments, etc.).",
                    "Determine the correct financial formula to use based on the question (Simple/Compound Interest, Annuity, etc.).",
                    "Ensure time periods and interest rates are consistent (e.g., annual rate with time in years).",
                    "Substitute values into the formula and calculate.",
                ]
            )
        elif math_type == "statistics":
            steps.extend(
                [
                    "Identify the data set (sample or population).",
                    "Determine the specific statistical measure required.",
                    "Apply the correct formula for that measure.",
                    "Perform the calculations accurately.",
                ]
            )
        else:
            steps.extend(
                [
                    "Break the problem down into smaller, manageable parts.",
                    "Perform necessary calculations carefully.",
                    "Review the steps and the final answer.",
                ]
            )
        return steps

    # --- Topic Modeling Methods (TF-IDF Implementation - Unmodified from v9) ---
    def _calculate_corpus_idf(
        self, documents: List[str], domain: str
    ) -> Dict[str, float]:
        num_documents = len(documents)
        if num_documents == 0:
            return {}
        df = Counter()
        tokenized_docs = []
        for doc in documents:
            tokens = self._tokenize(doc)
            filtered_tokens = {
                word
                for word in tokens
                if len(word) >= self.min_keyword_length and word not in self.stop_words
            }
            df.update(filtered_tokens)
            tokenized_docs.append(filtered_tokens)
        idf = {
            term: math.log(num_documents / (count + 1)) + 1
            for term, count in df.items()
        }
        return idf

    def _extract_topic_signature_tfidf(
        self, tfidf_vector: Dict[str, float]
    ) -> List[List[str]]:
        num_words = self.topic_config.get("num_words", 10)
        if not tfidf_vector:
            return []
        sorted_terms = sorted(
            tfidf_vector.items(), key=lambda item: item[1], reverse=True
        )
        top_keywords = [term for term, score in sorted_terms[:num_words]]
        return [top_keywords] if top_keywords else []

    def _compare_topic_signatures_cosine(
        self, vec1: Dict[str, float], vec2: Dict[str, float]
    ) -> Dict[str, Any]:
        similarity = self._cosine_similarity(vec1, vec2)
        overlap_score = similarity
        missing_keywords = []
        threshold = 0.1  # Threshold for 'high' TF-IDF score in target
        target_high_tfidf = {term for term, score in vec2.items() if score > threshold}
        gen_terms = set(vec1.keys())
        missing_keywords = list(target_high_tfidf - gen_terms)[:5]
        comparison = {
            "overlap_score": overlap_score,
            "missing_topic_keywords": missing_keywords,
        }
        return comparison

    def _get_topic_guidance_tfidf(
        self, topic_comparison: Dict[str, Any], domain: str
    ) -> Optional[str]:
        overlap_score = topic_comparison.get("overlap_score", 0.0)
        missing_keywords = topic_comparison.get("missing_topic_keywords", [])
        if overlap_score < 0.4 and missing_keywords:
            missing_keywords_sample = ", ".join(f"'{k}'" for k in missing_keywords[:3])
            return f"Your response seems to diverge conceptually from the target. Consider addressing concepts related to: {missing_keywords_sample}."
        elif overlap_score < 0.6:  # Moderate divergence
            return "Ensure your reasoning covers all the key aspects mentioned in the target or problem description more thoroughly."
        return None

    # --- Adaptive Loss Calculation Methods (MODIFIED) ---
    def _calculate_loss(
        self,
        gen_analysis: Dict,
        target_analysis: Dict,
        gen_final_answer: Optional[str],  # Answer extracted potentially using tags
        target_final_answer: Optional[str],  # Answer extracted potentially using tags
        math_analysis: Dict,
        topic_comparison: Dict,
        domain: str,
    ) -> Dict[str, float]:
        """Calculate weighted loss metrics, considering tag enforcement."""
        loss_metrics = {
            "structure": 0.0,
            "content": 0.0,
            "math": 0.0,
            "topic": 0.0,
            "guidance": 0.0,
        }
        raw_losses = {
            "structure": 0.0,
            "content": 0.0,
            "math": 0.0,
            "topic": 0.0,
            "guidance": 0.0,
        }

        # 1. Structure Loss (0-1) - Includes Tag Penalty if enforced
        penalty = 0.0
        # --- Tag Enforcement Penalty ---
        if self.enforce_tags:
            tag_penalty = 0.0
            if (
                target_analysis["has_thinking_tag"]
                and not gen_analysis["has_thinking_tag"]
            ):
                tag_penalty += self.tag_penalty_factor
            if target_analysis["has_answer_tag"] and not gen_analysis["has_answer_tag"]:
                tag_penalty += self.tag_penalty_factor
            penalty += min(
                tag_penalty, 1.0
            )  # Cap tag penalty at 1.0 before adding others

        # --- Original Structure Penalties ---
        if target_analysis["step_count"] > 1 and gen_analysis["step_count"] == 0:
            penalty += 0.7 * (
                1.0 - penalty
            )  # Reduce effect if already penalized by tags
        elif target_analysis["step_count"] > 0:
            penalty += (
                min(
                    1.0,
                    abs(gen_analysis["step_count"] - target_analysis["step_count"])
                    / target_analysis["step_count"],
                )
                * 0.4
                * (1.0 - penalty)
            )
        if (
            target_analysis["has_minimal_reasoning"]
            and not gen_analysis["has_minimal_reasoning"]
        ):
            penalty += 0.6 * (1.0 - penalty)
        if gen_analysis["has_uncertainty"]:
            penalty += 0.2 * (1.0 - penalty)
        if domain in self.reasoning_patterns and not gen_analysis.get(
            "detected_domain_pattern", False
        ):
            penalty += 0.3 * (1.0 - penalty)

        raw_losses["structure"] = min(penalty, 1.0)  # Final cap

        # 2. Content Loss (0-1) - Uses tagged answer if enforced and available
        penalty = 0.0
        target_answer_to_compare = (
            target_final_answer  # Use the potentially tag-extracted answer
        )
        gen_answer_to_compare = gen_final_answer

        if (
            target_answer_to_compare is not None
        ):  # Only penalize if target answer exists
            if gen_answer_to_compare is None:
                penalty += 0.9  # Heavy penalty for missing answer when expected
            else:
                # Simple normalized string comparison (case-insensitive, whitespace normalized)
                norm_gen = " ".join(gen_answer_to_compare.strip().lower().split())
                norm_target = " ".join(target_answer_to_compare.strip().lower().split())
                if norm_gen != norm_target:
                    penalty += 1.0  # Max penalty for wrong answer content

        # Keyword gap penalty (applied to thinking content if available and tag enforced, else full text)
        gen_text_for_keywords = (
            gen_analysis["thinking_content"]
            if (self.enforce_tags and gen_analysis["has_thinking_tag"])
            else gen_analysis["text"]
        )
        target_text_for_keywords = (
            target_analysis["thinking_content"]
            if (self.enforce_tags and target_analysis["has_thinking_tag"])
            else target_analysis["text"]
        )

        # Ensure texts are not None before checking keywords
        gen_text_for_keywords = gen_text_for_keywords or ""
        target_text_for_keywords = target_text_for_keywords or ""

        missing_keywords = self._check_keyword_gap(
            gen_text_for_keywords, target_text_for_keywords, domain
        )
        penalty += 0.05 * len(missing_keywords)

        raw_losses["content"] = min(penalty, 1.0)

        # 3. Math Loss (0-1) - Uses potentially tag-extracted answer for check
        if math_analysis.get("is_math_problem", False):
            penalty = 0.0
            # Check if answers exist and differ
            if (
                target_answer_to_compare is not None
                and gen_answer_to_compare is not None
            ):
                norm_gen_math = " ".join(gen_answer_to_compare.strip().lower().split())
                norm_target_math = " ".join(
                    target_answer_to_compare.strip().lower().split()
                )
                if norm_gen_math != norm_target_math:
                    penalty = 1.0  # Max penalty if final answers differ
            elif target_answer_to_compare is not None and gen_answer_to_compare is None:
                penalty = 1.0  # Max penalty if answer is missing
            # Penalize lack of steps if target has them
            elif target_analysis["step_count"] > 1 and gen_analysis["step_count"] < 1:
                penalty = max(
                    penalty, 0.4
                )  # Penalize missing steps even if answer is somehow correct/missing

            raw_losses["math"] = penalty

        # 4. Topic Loss (0-1) - Compare thinking content if available and enforced, else full text
        gen_text_for_topic = (
            gen_analysis["thinking_content"]
            if (self.enforce_tags and gen_analysis["has_thinking_tag"])
            else gen_analysis["text"]
        )
        target_text_for_topic = (
            target_analysis["thinking_content"]
            if (self.enforce_tags and target_analysis["has_thinking_tag"])
            else target_analysis["text"]
        )

        # Ensure texts are not None before calculating TF-IDF
        gen_text_for_topic = gen_text_for_topic or ""
        target_text_for_topic = target_text_for_topic or ""

        # Recalculate TF-IDF specifically for the chosen parts
        corpus_docs_topic = [gen_text_for_topic, target_text_for_topic]
        corpus_tokens_topic = [self._tokenize(doc) for doc in corpus_docs_topic]
        idf_scores_topic = self._calculate_idf(corpus_tokens_topic)
        gen_tfidf_vec_topic = self._calculate_tfidf_vector_manual(
            gen_text_for_topic, idf_scores_topic, domain
        )
        target_tfidf_vec_topic = self._calculate_tfidf_vector_manual(
            target_text_for_topic, idf_scores_topic, domain
        )
        topic_comparison_specific = self._compare_topic_signatures_cosine(
            gen_tfidf_vec_topic, target_tfidf_vec_topic
        )

        overlap_score = topic_comparison_specific.get(
            "overlap_score", 0.0
        )  # Cosine similarity
        raw_losses["topic"] = max(0.0, 1.0 - overlap_score)

        # 5. Guidance Effectiveness Loss (0-1) - Unmodified
        penalty = 0.5  # Default penalty
        if self._last_guidance_type:
            effectiveness_data = self.guidance_effectiveness.get(
                self._last_guidance_type
            )
            if effectiveness_data and effectiveness_data["scores"]:
                avg_effectiveness_score = effectiveness_data[
                    "avg"
                ]  # Score is 0-1 (1=good)
                penalty = max(0.0, 1.0 - avg_effectiveness_score)
        raw_losses["guidance"] = penalty

        # Apply weights
        weighted_losses = {
            f"{k}_loss": raw_losses[k] * self.loss_weights.get(f"{k}_weight", 1.0)
            for k in raw_losses
        }
        return weighted_losses

    def _evaluate_correctness(self, loss_metrics: Dict[str, float]) -> float:
        """Aggregates weighted loss metrics into a single score."""
        # (Implementation from v8 - unmodified)
        total_loss = sum(loss_metrics.values())
        return total_loss

    # --- Anti-Forgetting Methods (Unmodified from v9) ---
    def _generate_math_problem(
        self, difficulty: Optional[str] = None, domain: Optional[str] = None
    ) -> Tuple[Optional[str], Optional[str]]:
        if not self.math_problem_library:
            return None, None
        if not domain:
            domain = random.choice(
                self.anti_forgetting_config.get("domains", ["arithmetic"])
            )
        chosen_difficulty = difficulty
        if not chosen_difficulty:
            rand_val = random.random()
            cumulative_prob = 0.0
            dist = self.anti_forgetting_config.get(
                "difficulty_distribution", {"medium": 1.0}
            )
            total_prob = sum(dist.values())
            if not math.isclose(total_prob, 1.0) and total_prob > 0:
                dist = {k: v / total_prob for k, v in dist.items()}
            elif total_prob == 0:
                dist = {"medium": 1.0}
            sorted_dist = sorted(dist.items(), key=lambda item: item[1], reverse=True)
            for diff, prob in sorted_dist:
                cumulative_prob += prob
                if rand_val <= cumulative_prob:
                    chosen_difficulty = diff
                    break
            if not chosen_difficulty:
                chosen_difficulty = "medium"
        return self.math_problem_library.get_problem(
            domain=domain, difficulty=chosen_difficulty
        )

    # --- Performance Tracking (Unmodified from v9) ---
    def _update_guidance_effectiveness(
        self, guidance_type: Optional[str], current_total_loss: float
    ):
        if guidance_type and self._last_total_loss is not None:
            loss_reduction = self._last_total_loss - current_total_loss
            effectiveness_score = 1.0 if loss_reduction > 0.01 else 0.0
            data = self.guidance_effectiveness[guidance_type]
            data["scores"].append(effectiveness_score)
            data["avg"] = sum(data["scores"]) / len(data["scores"])

    def get_guidance_statistics(self) -> Dict[str, Any]:
        stats = {}
        for guidance_type, data in self.guidance_effectiveness.items():
            if data["scores"]:
                stats[str(guidance_type)] = {
                    "count": len(data["scores"]),
                    "average_effectiveness": round(data["avg"], 3),
                    "effectiveness_trend": data["scores"][-10:],
                }
        return stats

    def generate_guidance(
        self,
        current_step: int,
        conversation_messages: List[Dict[str, str]],
        last_generation: str,
        original_target_text: str,
        domain: str = "general"
        # --- MODIFIED: Return signature ---
    ) -> Tuple[
        Optional[str], str, float
    ]:  # Returns (Guidance Text, Loss Report String, Raw Total Loss)
        """
        Generate strategic guidance, an adaptive loss report, and the raw total loss value.
        Optionally enforces tags. (v11 - Return Loss)
        """
        guidance: Optional[str] = None
        current_guidance_type: Optional[str] = None
        total_loss: float = 1.0  # Default high loss in case of early exit

        # --- 1. Input Validation & Initial Analysis ---
        if not last_generation:
            guidance = "It seems the generation failed. Please try generating a response, showing your reasoning steps."
            current_guidance_type = "generation_failed"
            # Assume max loss for all components when generation fails
            # Calculate raw losses before weighting
            raw_losses = {
                k: 1.0 for k in ["structure", "content", "math", "topic", "guidance"]
            }
            loss_metrics = {
                f"{k}_loss": raw_losses[k] * self.loss_weights.get(f"{k}_weight", 1.0)
                for k in raw_losses
            }
            total_loss = sum(
                loss_metrics.values()
            )  # Use weighted sum for internal logic consistency
            self._update_guidance_effectiveness(self._last_guidance_type, total_loss)
            self._last_total_loss = total_loss
            self._last_gen_reasoning_length = 0
            loss_report = f"Adaptive Loss Report (Generation Failed): Total={total_loss:.2f} ({'; '.join(f'{k}={v:.2f}' for k,v in loss_metrics.items())})"
            self._last_guidance_type = current_guidance_type
            self.loss_history.append(total_loss)
            # --- MODIFIED: Return default total_loss ---
            return guidance, loss_report, total_loss  # Return high loss

        gen_analysis = self._analyze_cot_structure(last_generation, domain)
        target_analysis = self._analyze_cot_structure(original_target_text, domain)
        gen_final_answer = self._extract_final_answer(last_generation, gen_analysis)
        target_final_answer = self._extract_final_answer(
            original_target_text, target_analysis
        )
        inferred_stage = self._infer_reasoning_stage(
            gen_analysis, conversation_messages
        )
        last_user_message = next(
            (
                msg["content"]
                for msg in reversed(conversation_messages)
                if msg.get("role") == "user"
            ),
            "",
        )
        math_analysis = (
            self._detect_math_content(last_user_message)
            if last_user_message
            else {"is_math_problem": False}
        )
        is_math_problem = math_analysis.get("is_math_problem", False)
        effective_domain = "math" if is_math_problem else domain
        # Topic modeling (preliminary)
        gen_text_topic_prelim = (
            gen_analysis["thinking_content"]
            if (self.enforce_tags and gen_analysis["has_thinking_tag"])
            else gen_analysis["text"]
        )
        target_text_topic_prelim = (
            target_analysis["thinking_content"]
            if (self.enforce_tags and target_analysis["has_thinking_tag"])
            else target_analysis["text"]
        )
        gen_text_topic_prelim = gen_text_topic_prelim or ""
        target_text_topic_prelim = target_text_topic_prelim or ""
        corpus_docs_prelim = [gen_text_topic_prelim, target_text_topic_prelim]
        corpus_tokens_prelim = [self._tokenize(doc) for doc in corpus_docs_prelim]
        idf_scores_prelim = self._calculate_idf(corpus_tokens_prelim)
        gen_tfidf_vec_prelim = self._calculate_tfidf_vector_manual(
            gen_text_topic_prelim, idf_scores_prelim, effective_domain
        )
        target_tfidf_vec_prelim = self._calculate_tfidf_vector_manual(
            target_text_topic_prelim, idf_scores_prelim, effective_domain
        )
        topic_comparison_prelim = self._compare_topic_signatures_cosine(
            gen_tfidf_vec_prelim, target_tfidf_vec_prelim
        )

        # --- Calculate Loss ---
        loss_metrics = self._calculate_loss(
            gen_analysis=gen_analysis,
            target_analysis=target_analysis,
            gen_final_answer=gen_final_answer,
            target_final_answer=target_final_answer,
            math_analysis=math_analysis,
            topic_comparison=topic_comparison_prelim,
            domain=effective_domain,
        )
        # --- Store the calculated total loss ---
        total_loss = self._evaluate_correctness(
            loss_metrics
        )  # This is the weighted sum
        self.loss_history.append(total_loss)
        # --- Update Effectiveness ---
        self._update_guidance_effectiveness(self._last_guidance_type, total_loss)

        # --- 5. Determine Next Step Target & Struggle Detection ---
        next_step_target: str = original_target_text
        is_struggling = False
        struggle_reason = ""
        struggle_type = "none"
        struggle_threshold_factor = 0.7
        # Tag check
        if self.enforce_tags:
            if (
                target_analysis["has_thinking_tag"]
                and not gen_analysis["has_thinking_tag"]
            ):
                guidance = f"Please structure your response using `{self.thinking_open}...{self.thinking_close}` tags to clearly separate your reasoning process."
                current_guidance_type = "missing_thinking_tag"
                is_struggling = True
                struggle_type = "missing_tag"
            elif (
                target_analysis["has_answer_tag"]
                and not gen_analysis["has_answer_tag"]
                and guidance is None
            ):
                guidance = f"Please clearly mark your final answer using `{self.answer_open}...{self.answer_close}` tags."
                current_guidance_type = "missing_answer_tag"
                is_struggling = True
                struggle_type = "missing_tag"
        # Other struggle checks
        if guidance is None:
            if self._detect_repetition(last_generation):
                struggle_reason = "It looks like the reasoning is repeating itself. and is very short. can you go deeper try thinking about thinking?"
                struggle_type = "repetition"
                is_struggling = True
            elif (
                gen_analysis["has_uncertainty"]
                and not gen_analysis["has_minimal_reasoning"]
            ):
                struggle_reason = "It seems there's some confusion or difficulty expressing the reasoning.Use markdown multi cols table, to layout whats on your mind?"
                struggle_type = "uncertainty_stuck"
                is_struggling = True
            elif loss_metrics.get(
                "structure_loss", 0.0
            ) > struggle_threshold_factor * self.loss_weights.get(
                "structure_weight", 1.0
            ):
                if not (self.enforce_tags and struggle_type == "missing_tag"):
                    struggle_reason = "The reasoning structure seems significantly flawed or incomplete.Redo and go depper, try converting each step and sub step to a stage and then figure out steps and sub steps under each (use markdown multi cols table)"
                    struggle_type = "structure_issue"
                    is_struggling = True
            elif loss_metrics.get(
                "content_loss", 0.0
            ) > struggle_threshold_factor * self.loss_weights.get(
                "content_weight", 1.0
            ):
                struggle_reason = "The content or final answer seems significantly incorrect or off-topic."
                struggle_type = "content_issue"
                is_struggling = True
            elif is_math_problem and loss_metrics.get(
                "math_loss", 0.0
            ) > struggle_threshold_factor * self.loss_weights.get("math_weight", 1.0):
                struggle_reason = (
                    "The mathematical result or process appears incorrect."
                )
                struggle_type = "math_error"
                is_struggling = True

            if is_struggling:
                current_guidance_type = f"handle_{struggle_type}"
                guidance = struggle_reason
                math_steps = None
                if is_math_problem:
                    math_steps = self._decompose_math_problem(
                        last_user_message, math_analysis
                    )
                    if math_steps:
                        first_step_focus = math_steps[0]
                        guidance += f" Let's approach this math problem methodically. Start by focusing on the first logical step (use markdown multi cols table): '{first_step_focus}'"
                        next_step_target = f"Focus on: {first_step_focus}"  # Update target if decomposing
                    else:
                        is_math_problem = False  # Treat as general if decomp fails

                if not is_math_problem or not math_steps:
                    # Use thinking content for breakdown if available and enforced
                    target_text_for_breakdown = (
                        target_analysis["thinking_content"]
                        if (self.enforce_tags and target_analysis["has_thinking_tag"])
                        else original_target_text
                    )
                    target_text_for_breakdown = (
                        target_text_for_breakdown or original_target_text
                    )  # Fallback
                    target_breakdown = self._breakdown_target(target_text_for_breakdown)

                    if target_breakdown:
                        first_subgoal = target_breakdown[0]
                        guidance += f" Let's simplify. Try focusing *only* on this first part: '{first_subgoal}'"
                        next_step_target = first_subgoal  # Update target if decomposing
                    else:
                        guidance += (
                            " Let's try to restart the thinking process clearly."
                        )
                        next_step_target = original_target_text  # Reset target

                # Avoid redundant struggle guidance
                if (
                    current_guidance_type
                    and current_guidance_type == self._last_guidance_type
                ):
                    guidance = (
                        None  # Suppress if same struggle guidance given last time
                    )

        # --- 6. Generate Standard Guidance (If No Struggle/Tag Guidance Given) ---
        if guidance is None:
            progress_fraction = current_step / self.total_steps
            # Use appropriate text for keyword gap check based on tags/enforcement
            gen_text_kw = (
                gen_analysis["thinking_content"]
                if (self.enforce_tags and gen_analysis["has_thinking_tag"])
                else gen_analysis["text"]
            )
            target_text_kw = (
                target_analysis["thinking_content"]
                if (self.enforce_tags and target_analysis["has_thinking_tag"])
                else target_analysis["text"]
            )
            gen_text_kw = gen_text_kw or ""
            target_text_kw = target_text_kw or ""
            missing_keywords = self._check_keyword_gap(
                gen_text_kw, target_text_kw, effective_domain
            )

            # Use preliminary topic comparison results
            topic_guidance = self._get_topic_guidance_tfidf(
                topic_comparison_prelim, effective_domain
            )

            # --- Determine guidance based on priority: High Loss > Stage/Content Issues > Generic ---
            potential_guidances = []  # (priority, type, text)

            # 6a. High Loss Components (Highest Priority)
            loss_threshold = 0.5  # Threshold for triggering specific loss guidance
            if loss_metrics.get(
                "content_loss", 0.0
            ) > loss_threshold * self.loss_weights.get("content_weight", 1.0):
                g_text = "Content/answer seems incorrect. Please review your reasoning and calculations carefully."
                if missing_keywords:
                    g_text = f"Content/answer seems incorrect. Did you consider '{missing_keywords[0]}'? Review your steps."
                potential_guidances.append((10, "review_steps_for_error", g_text))
            if loss_metrics.get(
                "math_loss", 0.0
            ) > loss_threshold * self.loss_weights.get("math_weight", 1.0):
                math_type = math_analysis.get("math_type", "general_math")
                templates = self.math_guidance_templates.get(
                    math_type, self.math_guidance_templates["general_math"]
                )
                check_templates = [
                    t
                    for t in templates
                    if "check" in t.lower()
                    or "verify" in t.lower()
                    or "step" in t.lower()
                ]
                potential_guidances.append(
                    (
                        10,
                        "math_final_verification",
                        f"There might be an error in the math ({math_type}). {random.choice(check_templates if check_templates else templates)}",
                    )
                )
            # Only trigger structure guidance if not already handled by tag check or struggle logic
            if (
                loss_metrics.get("structure_loss", 0.0)
                > loss_threshold * self.loss_weights.get("structure_weight", 1.0)
                and struggle_type != "structure_issue"
                and struggle_type != "missing_tag"
            ):
                potential_guidances.append(
                    (
                        9,
                        "initiate_cot_planning",
                        "The reasoning structure seems unclear or incomplete. Try outlining steps or thinking step-by-step.",
                    )
                )
            if (
                loss_metrics.get("topic_loss", 0.0)
                > loss_threshold * self.loss_weights.get("topic_weight", 1.0)
                and topic_guidance
            ):
                potential_guidances.append(
                    (8, "increase_detail_concepts", topic_guidance)
                )

            # 6b. Stage/Content Issues (Medium Priority)
            if not potential_guidances:
                if inferred_stage == self.STAGE_INITIATION:
                    g_type = "initiate_cot_planning"
                    g_text = (
                        "Please start by thinking step-by-step or outlining a plan."
                    )
                    if self.enforce_tags:
                        g_text += f" Remember to use `{self.thinking_open}...{self.thinking_close}` for your reasoning."
                    if is_math_problem:
                        math_type = math_analysis.get("math_type", "general_math")
                        g_text = f"This looks like a {math_type} problem. {random.choice(self.math_guidance_templates.get(math_type, self.math_guidance_templates['general_math']))}"
                    potential_guidances.append((7, g_type, g_text))
                elif missing_keywords or (
                    gen_analysis["reasoning_length_chars"]
                    < target_analysis["reasoning_length_chars"] * 0.5
                    and target_analysis["has_minimal_reasoning"]
                ):
                    g_type = "increase_detail_concepts"
                    g_text = "Can you elaborate more on your reasoning steps?"
                    if self.enforce_tags and gen_analysis["has_thinking_tag"]:
                        g_text += f" Expand the details within the `{self.thinking_open}...{self.thinking_close}` block."
                    if is_math_problem:
                        math_type = math_analysis.get("math_type", "general_math")
                        detail_templates = [
                            t
                            for t in self.math_guidance_templates.get(math_type, [])
                            if "step" in t.lower() or "detail" in t.lower()
                        ]
                        g_text = f"Reasoning seems brief for this {math_type} problem. {random.choice(detail_templates if detail_templates else ['Show more steps.'])}"
                    if missing_keywords:
                        g_text += (
                            f" Perhaps mention how '{missing_keywords[0]}' fits in?"
                        )
                    potential_guidances.append((6, g_type, g_text))
                elif (
                    effective_domain in self.reasoning_patterns
                    and not gen_analysis.get("detected_domain_pattern", False)
                ):
                    potential_guidances.append(
                        (
                            6,
                            "apply_domain_pattern",
                            f"Try structuring your reasoning using typical patterns for {effective_domain} problems (e.g., state premises/formula, show steps, conclude).",
                        )
                    )

            # 6c. Generic Stage-Based Guidance (Lowest Priority) - use Adaptive Memory
            if not potential_guidances:
                guidance_options = []  # List of (type, text)
                # Populate options based on stage
                if (
                    inferred_stage == self.STAGE_DECOMPOSITION
                    and progress_fraction < 0.7
                ):
                    guidance_options.append(
                        (
                            "next_step_elaboration",
                            "You've started outlining steps. What's the next logical step? Elaborate on it.",
                        )
                    )
                elif inferred_stage == self.STAGE_REASONING and progress_fraction < 0.8:
                    choices = [
                        "Are there any alternative approaches?",
                        "Double-check the logic of recent steps.",
                        "What assumptions are you making?",
                    ]
                    guidance_options.append(
                        ("mid_reasoning_check", random.choice(choices))
                    )
                elif inferred_stage == self.STAGE_VERIFICATION or (
                    gen_analysis["has_minimal_reasoning"] and progress_fraction >= 0.7
                ):
                    choices = [
                        "Review your entire reasoning. Does the conclusion follow?",
                        "Are there any edge cases?",
                        "Could a skeptic find flaws?",
                    ]
                    guidance_options.append(
                        ("final_verification_critique", random.choice(choices))
                    )

                if guidance_options:
                    # Epsilon-Greedy Selection
                    epsilon = self.adaptive_memory_config.get("epsilon", 0.1)
                    chosen_option = None
                    if random.random() < epsilon:
                        chosen_option = random.choice(guidance_options)  # Explore
                    else:  # Exploit
                        best_avg_eff = -1.0
                        best_choice = None
                        context_key = (
                            effective_domain,
                            inferred_stage,
                            struggle_type,
                        )  # Context key
                        # Simple exploit: use global average for the type for now
                        for g_type, g_text in guidance_options:
                            hist_data = self.guidance_effectiveness.get(g_type)
                            avg_eff = (
                                hist_data["avg"] if hist_data else 0.5
                            )  # Default effectiveness
                            if avg_eff >= best_avg_eff:
                                best_avg_eff = avg_eff
                                best_choice = (g_type, g_text)
                        chosen_option = (
                            best_choice
                            if best_choice
                            else random.choice(guidance_options)
                        )  # Fallback

                    if chosen_option:
                        potential_guidances.append(
                            (5, chosen_option[0], chosen_option[1])
                        )

            # Select the highest priority guidance generated
            if potential_guidances:
                potential_guidances.sort(key=lambda x: x[0], reverse=True)
                current_guidance_type = potential_guidances[0][1]
                guidance = potential_guidances[0][2]

        # --- 7. Final Formatting, Effectiveness Tracking & Return ---
        current_reasoning_length = gen_analysis.get("reasoning_length_chars", 0)

        # Avoid redundant generic guidance
        critical_guidance_types = {
            "review_steps_for_error",
            "missing_thinking_tag",
            "missing_answer_tag",
            "handle_math_error",
            "handle_content_issue",
            "handle_repetition",
            "handle_uncertainty_stuck",
            "handle_structure_issue",
            "initiate_cot_planning",
        }
        if (
            current_guidance_type
            and current_guidance_type == self._last_guidance_type
            and current_guidance_type not in critical_guidance_types
        ):
            guidance = None
            current_guidance_type = None

        # Record guidance type and update state
        self._last_guidance_type = current_guidance_type
        self._last_gen_reasoning_length = current_reasoning_length
        self._last_total_loss = total_loss

        # Format guidance string
        formatted_guidance = f"{guidance}" if guidance else None

        # Prepare Loss Report String
        loss_report_items = [
            f"{k.replace('_loss', '')}={v:.2f}" for k, v in loss_metrics.items()
        ]
        loss_report = f"Adaptive Loss Report: Total={total_loss:.2f} ({'; '.join(loss_report_items)})"

        # --- MODIFIED: Return guidance, report, AND the raw total loss ---
        return formatted_guidance, loss_report, total_loss

    # --- Main Interface Method (MODIFIED) ---
    def generate_guidance_old(
        self,
        current_step: int,
        conversation_messages: List[Dict[str, str]],
        last_generation: str,
        original_target_text: str,
        domain: str = "general",
    ) -> Tuple[Optional[str], str]:  # Returns (Guidance Text, Loss Report String)
        """
        Generate strategic guidance and an adaptive loss report, optionally enforcing tags. (v10)
        """
        guidance: Optional[str] = None
        current_guidance_type: Optional[str] = None

        # --- 1. Input Validation & Initial Analysis (Includes Tag Extraction) ---
        if not last_generation:
            # Handle generation failure (unmodified)
            guidance = "It seems the generation failed. Please try generating a response, showing your reasoning steps."
            current_guidance_type = "generation_failed"
            # Assume max loss for all components when generation fails
            loss_metrics = {
                f"{k}_loss": 1.0 * self.loss_weights.get(f"{k}_weight", 1.0)
                for k in ["structure", "content", "math", "topic", "guidance"]
            }
            total_loss = sum(loss_metrics.values())
            self._update_guidance_effectiveness(self._last_guidance_type, total_loss)
            self._last_total_loss = total_loss
            self._last_gen_reasoning_length = 0
            loss_report = f"Adaptive Loss Report (Generation Failed): Total={total_loss:.2f} ({'; '.join(f'{k}={v:.2f}' for k,v in loss_metrics.items())})"
            self._last_guidance_type = current_guidance_type
            self.loss_history.append(total_loss)
            return guidance, loss_report

        # Perform analyses (now includes tag info)
        gen_analysis = self._analyze_cot_structure(last_generation, domain)
        target_analysis = self._analyze_cot_structure(original_target_text, domain)

        # Extract final answers (prioritizing tags if enforced)
        gen_final_answer = self._extract_final_answer(last_generation, gen_analysis)
        target_final_answer = self._extract_final_answer(
            original_target_text, target_analysis
        )

        inferred_stage = self._infer_reasoning_stage(
            gen_analysis, conversation_messages
        )
        last_user_message = next(
            (
                msg["content"]
                for msg in reversed(conversation_messages)
                if msg.get("role") == "user"
            ),
            "",
        )
        math_analysis = (
            self._detect_math_content(last_user_message)
            if last_user_message
            else {"is_math_problem": False}
        )
        is_math_problem = math_analysis.get("is_math_problem", False)
        effective_domain = "math" if is_math_problem else domain

        # --- 2. Topic Modeling (Uses appropriate text based on enforcement/tags) ---
        # Texts for topic modeling are selected within _calculate_loss now
        # We calculate a preliminary comparison here for potential guidance use later
        gen_text_topic_prelim = (
            gen_analysis["thinking_content"]
            if (self.enforce_tags and gen_analysis["has_thinking_tag"])
            else gen_analysis["text"]
        )
        target_text_topic_prelim = (
            target_analysis["thinking_content"]
            if (self.enforce_tags and target_analysis["has_thinking_tag"])
            else target_analysis["text"]
        )
        gen_text_topic_prelim = gen_text_topic_prelim or ""
        target_text_topic_prelim = target_text_topic_prelim or ""

        corpus_docs_prelim = [gen_text_topic_prelim, target_text_topic_prelim]
        corpus_tokens_prelim = [self._tokenize(doc) for doc in corpus_docs_prelim]
        idf_scores_prelim = self._calculate_idf(corpus_tokens_prelim)
        gen_tfidf_vec_prelim = self._calculate_tfidf_vector_manual(
            gen_text_topic_prelim, idf_scores_prelim, effective_domain
        )
        target_tfidf_vec_prelim = self._calculate_tfidf_vector_manual(
            target_text_topic_prelim, idf_scores_prelim, effective_domain
        )
        topic_comparison_prelim = self._compare_topic_signatures_cosine(
            gen_tfidf_vec_prelim, target_tfidf_vec_prelim
        )

        # --- 3. Calculate Adaptive Loss (Uses tag info) ---
        loss_metrics = self._calculate_loss(
            gen_analysis=gen_analysis,
            target_analysis=target_analysis,
            gen_final_answer=gen_final_answer,
            target_final_answer=target_final_answer,
            math_analysis=math_analysis,
            # Pass the preliminary topic comparison, loss will refine if needed
            topic_comparison=topic_comparison_prelim,
            domain=effective_domain,
        )
        total_loss = self._evaluate_correctness(loss_metrics)
        self.loss_history.append(total_loss)

        # --- 4. Update Effectiveness Tracking for *Previous* Guidance ---
        self._update_guidance_effectiveness(self._last_guidance_type, total_loss)

        # --- 5. Determine Next Step Target & Initial Guidance (Struggle/Tag Enforcement) ---
        next_step_target: str = original_target_text  # Default target
        is_struggling = False
        struggle_reason = ""
        struggle_type = "none"
        struggle_threshold_factor = 0.7

        # --- 5a. Check for Missing Tags (Highest Priority if Enforced) ---
        if self.enforce_tags:
            if (
                target_analysis["has_thinking_tag"]
                and not gen_analysis["has_thinking_tag"]
            ):
                guidance = f"Please structure your response using `{self.thinking_open}...{self.thinking_close}` tags to clearly separate your reasoning process."
                current_guidance_type = "missing_thinking_tag"
                is_struggling = True  # Treat missing required tag as a struggle
                struggle_type = "missing_tag"
            elif (
                target_analysis["has_answer_tag"] and not gen_analysis["has_answer_tag"]
            ):
                # Give this slightly lower priority than missing thinking tag if both are missing
                if guidance is None:
                    guidance = f"Please clearly mark your final answer using `{self.answer_open}...{self.answer_close}` tags."
                    current_guidance_type = "missing_answer_tag"
                    is_struggling = True
                    struggle_type = "missing_tag"

        # --- 5b. Check for Other Struggles (Only if tag guidance wasn't triggered) ---
        if guidance is None:
            if self._detect_repetition(last_generation):
                struggle_reason = "It looks like the reasoning is repeating itself."
                struggle_type = "repetition"
                is_struggling = True
            elif (
                gen_analysis["has_uncertainty"]
                and not gen_analysis["has_minimal_reasoning"]
            ):
                struggle_reason = "It seems there's some confusion or difficulty expressing the reasoning."
                struggle_type = "uncertainty_stuck"
                is_struggling = True
            elif loss_metrics.get(
                "structure_loss", 0.0
            ) > struggle_threshold_factor * self.loss_weights.get(
                "structure_weight", 1.0
            ):
                # Avoid triggering structure loss guidance if it was primarily due to missing tags already handled
                if not (self.enforce_tags and struggle_type == "missing_tag"):
                    struggle_reason = "The reasoning structure seems significantly flawed or incomplete."
                    struggle_type = "structure_issue"
                    is_struggling = True
            elif loss_metrics.get(
                "content_loss", 0.0
            ) > struggle_threshold_factor * self.loss_weights.get(
                "content_weight", 1.0
            ):
                struggle_reason = "The content or final answer seems significantly incorrect or off-topic."
                struggle_type = "content_issue"
                is_struggling = True
            elif is_math_problem and loss_metrics.get(
                "math_loss", 0.0
            ) > struggle_threshold_factor * self.loss_weights.get("math_weight", 1.0):
                struggle_reason = (
                    "The mathematical result or process appears incorrect."
                )
                struggle_type = "math_error"
                is_struggling = True

            if is_struggling:
                current_guidance_type = (
                    f"handle_{struggle_type}"  # Use specific handle type
                )
                guidance = struggle_reason
                math_steps = None
                if is_math_problem:
                    math_steps = self._decompose_math_problem(
                        last_user_message, math_analysis
                    )
                    if math_steps:
                        first_step_focus = math_steps[0]
                        guidance += f" Let's approach this math problem methodically. Start by focusing on the first logical step: '{first_step_focus}'"
                        next_step_target = f"Focus on: {first_step_focus}"  # Update target if decomposing
                    else:
                        is_math_problem = False  # Treat as general if decomp fails

                if not is_math_problem or not math_steps:
                    # Use thinking content for breakdown if available and enforced
                    target_text_for_breakdown = (
                        target_analysis["thinking_content"]
                        if (self.enforce_tags and target_analysis["has_thinking_tag"])
                        else original_target_text
                    )
                    target_text_for_breakdown = (
                        target_text_for_breakdown or original_target_text
                    )  # Fallback
                    target_breakdown = self._breakdown_target(target_text_for_breakdown)

                    if target_breakdown:
                        first_subgoal = target_breakdown[0]
                        guidance += f" Let's simplify. Try focusing *only* on this first part: '{first_subgoal}'"
                        next_step_target = first_subgoal  # Update target if decomposing
                    else:
                        guidance += (
                            " Let's try to restart the thinking process clearly."
                        )
                        next_step_target = original_target_text  # Reset target

                # Avoid redundant struggle guidance
                if (
                    current_guidance_type
                    and current_guidance_type == self._last_guidance_type
                ):
                    guidance = (
                        None  # Suppress if same struggle guidance given last time
                    )

        # --- 6. Generate Standard Guidance (If No Struggle/Tag Guidance Given) ---
        if guidance is None:
            progress_fraction = current_step / self.total_steps
            # Use appropriate text for keyword gap check based on tags/enforcement
            gen_text_kw = (
                gen_analysis["thinking_content"]
                if (self.enforce_tags and gen_analysis["has_thinking_tag"])
                else gen_analysis["text"]
            )
            target_text_kw = (
                target_analysis["thinking_content"]
                if (self.enforce_tags and target_analysis["has_thinking_tag"])
                else target_analysis["text"]
            )
            gen_text_kw = gen_text_kw or ""
            target_text_kw = target_text_kw or ""
            missing_keywords = self._check_keyword_gap(
                gen_text_kw, target_text_kw, effective_domain
            )

            # Use preliminary topic comparison results
            topic_guidance = self._get_topic_guidance_tfidf(
                topic_comparison_prelim, effective_domain
            )

            # --- Determine guidance based on priority: High Loss > Stage/Content Issues > Generic ---
            potential_guidances = []  # (priority, type, text)

            # 6a. High Loss Components (Highest Priority)
            loss_threshold = 0.5  # Threshold for triggering specific loss guidance
            if loss_metrics.get(
                "content_loss", 0.0
            ) > loss_threshold * self.loss_weights.get("content_weight", 1.0):
                g_text = "Content/answer seems incorrect. Please review your reasoning and calculations carefully."
                if missing_keywords:
                    g_text = f"Content/answer seems incorrect. Did you consider '{missing_keywords[0]}'? Review your steps."
                potential_guidances.append((10, "review_steps_for_error", g_text))
            if loss_metrics.get(
                "math_loss", 0.0
            ) > loss_threshold * self.loss_weights.get("math_weight", 1.0):
                math_type = math_analysis.get("math_type", "general_math")
                templates = self.math_guidance_templates.get(
                    math_type, self.math_guidance_templates["general_math"]
                )
                check_templates = [
                    t
                    for t in templates
                    if "check" in t.lower()
                    or "verify" in t.lower()
                    or "step" in t.lower()
                ]
                potential_guidances.append(
                    (
                        10,
                        "math_final_verification",
                        f"There might be an error in the math ({math_type}). {random.choice(check_templates if check_templates else templates)}",
                    )
                )
            # Only trigger structure guidance if not already handled by tag check or struggle logic
            if (
                loss_metrics.get("structure_loss", 0.0)
                > loss_threshold * self.loss_weights.get("structure_weight", 1.0)
                and struggle_type != "structure_issue"
                and struggle_type != "missing_tag"
            ):
                potential_guidances.append(
                    (
                        9,
                        "initiate_cot_planning",
                        "The reasoning structure seems unclear or incomplete. Try outlining steps or thinking step-by-step.",
                    )
                )
            if (
                loss_metrics.get("topic_loss", 0.0)
                > loss_threshold * self.loss_weights.get("topic_weight", 1.0)
                and topic_guidance
            ):
                potential_guidances.append(
                    (8, "increase_detail_concepts", topic_guidance)
                )

            # 6b. Stage/Content Issues (Medium Priority)
            if not potential_guidances:
                if inferred_stage == self.STAGE_INITIATION:
                    g_type = "initiate_cot_planning"
                    g_text = (
                        "Please start by thinking step-by-step or outlining a plan."
                    )
                    if self.enforce_tags:
                        g_text += f" Remember to use `{self.thinking_open}...{self.thinking_close}` for your reasoning."
                    if is_math_problem:
                        math_type = math_analysis.get("math_type", "general_math")
                        g_text = f"This looks like a {math_type} problem. {random.choice(self.math_guidance_templates.get(math_type, self.math_guidance_templates['general_math']))}"
                    potential_guidances.append((7, g_type, g_text))
                elif missing_keywords or (
                    gen_analysis["reasoning_length_chars"]
                    < target_analysis["reasoning_length_chars"] * 0.5
                    and target_analysis["has_minimal_reasoning"]
                ):
                    g_type = "increase_detail_concepts"
                    g_text = "Can you elaborate more on your reasoning steps?"
                    if self.enforce_tags and gen_analysis["has_thinking_tag"]:
                        g_text += f" Expand the details within the `{self.thinking_open}...{self.thinking_close}` block."
                    if is_math_problem:
                        math_type = math_analysis.get("math_type", "general_math")
                        detail_templates = [
                            t
                            for t in self.math_guidance_templates.get(math_type, [])
                            if "step" in t.lower() or "detail" in t.lower()
                        ]
                        g_text = f"Reasoning seems brief for this {math_type} problem. {random.choice(detail_templates if detail_templates else ['Show more steps.'])}"
                    if missing_keywords:
                        g_text += (
                            f" Perhaps mention how '{missing_keywords[0]}' fits in?"
                        )
                    potential_guidances.append((6, g_type, g_text))
                elif (
                    effective_domain in self.reasoning_patterns
                    and not gen_analysis.get("detected_domain_pattern", False)
                ):
                    potential_guidances.append(
                        (
                            6,
                            "apply_domain_pattern",
                            f"Try structuring your reasoning using typical patterns for {effective_domain} problems (e.g., state premises/formula, show steps, conclude).",
                        )
                    )

            # 6c. Generic Stage-Based Guidance (Lowest Priority) - use Adaptive Memory
            if not potential_guidances:
                guidance_options = []  # List of (type, text)
                # Populate options based on stage
                if (
                    inferred_stage == self.STAGE_DECOMPOSITION
                    and progress_fraction < 0.7
                ):
                    guidance_options.append(
                        (
                            "next_step_elaboration",
                            "You've started outlining steps. What's the next logical step? Elaborate on it.",
                        )
                    )
                elif inferred_stage == self.STAGE_REASONING and progress_fraction < 0.8:
                    choices = [
                        "Are there any alternative approaches?",
                        "Double-check the logic of recent steps.",
                        "What assumptions are you making?",
                    ]
                    guidance_options.append(
                        ("mid_reasoning_check", random.choice(choices))
                    )
                elif inferred_stage == self.STAGE_VERIFICATION or (
                    gen_analysis["has_minimal_reasoning"] and progress_fraction >= 0.7
                ):
                    choices = [
                        "Review your entire reasoning. Does the conclusion follow?",
                        "Are there any edge cases?",
                        "Could a skeptic find flaws?",
                    ]
                    guidance_options.append(
                        ("final_verification_critique", random.choice(choices))
                    )

                if guidance_options:
                    # Epsilon-Greedy Selection (Unmodified logic)
                    epsilon = self.adaptive_memory_config.get("epsilon", 0.1)
                    chosen_option = None
                    if random.random() < epsilon:
                        chosen_option = random.choice(guidance_options)  # Explore
                    else:  # Exploit
                        best_avg_eff = -1.0
                        best_choice = None
                        context_key = (
                            effective_domain,
                            inferred_stage,
                            struggle_type,
                        )  # Context key
                        # Simple exploit: use global average for the type for now
                        for g_type, g_text in guidance_options:
                            hist_data = self.guidance_effectiveness.get(
                                g_type
                            )  # TODO: Use context_key? Needs better storage/retrieval
                            avg_eff = (
                                hist_data["avg"] if hist_data else 0.5
                            )  # Default effectiveness
                            if avg_eff >= best_avg_eff:
                                best_avg_eff = avg_eff
                                best_choice = (g_type, g_text)
                        chosen_option = (
                            best_choice
                            if best_choice
                            else random.choice(guidance_options)
                        )  # Fallback

                    if chosen_option:
                        potential_guidances.append(
                            (5, chosen_option[0], chosen_option[1])
                        )

            # Select the highest priority guidance generated
            if potential_guidances:
                potential_guidances.sort(key=lambda x: x[0], reverse=True)
                current_guidance_type = potential_guidances[0][1]
                guidance = potential_guidances[0][2]

        # --- 7. Final Formatting, Effectiveness Tracking & Return ---
        current_reasoning_length = gen_analysis.get("reasoning_length_chars", 0)

        # (Effectiveness tracking already updated in step 4)

        # Avoid redundant generic guidance (check against last *actual* guidance type)
        critical_guidance_types = {
            "review_steps_for_error",
            "handle_repetition",
            "handle_uncertainty_stuck",
            "handle_structure_issue",
            "handle_content_issue",
            "handle_math_error",
            "initiate_cot_planning",
            "missing_thinking_tag",
            "missing_answer_tag",  # Add new critical types
        }
        if (
            current_guidance_type
            and current_guidance_type == self._last_guidance_type
            and current_guidance_type not in critical_guidance_types
        ):
            guidance = None  # Suppress redundant non-critical guidance
            current_guidance_type = None  # Clear type if suppressed

        # Record the type of guidance *actually* being given now (could be None)
        self._last_guidance_type = current_guidance_type

        # Update state for next step's calculations
        self._last_gen_reasoning_length = current_reasoning_length
        self._last_total_loss = total_loss  # Store loss for next effectiveness calc

        # Format guidance string
        formatted_guidance = f"[CoT Guidance: {guidance}]" if guidance else None

        # Prepare Loss Report String
        loss_report_items = [
            f"{k.replace('_loss', '')}={v:.2f}" for k, v in loss_metrics.items()
        ]
        loss_report = f"Adaptive Loss Report: Total={total_loss:.2f} ({'; '.join(loss_report_items)})"

        # Return guidance text and the loss report string
        return formatted_guidance, loss_report


# --- Example Usage (MODIFIED) ---
if __name__ == "__main__":
    TOTAL_TRAINING_STEPS = 4

    # --- Configuration 1: Tag Enforcement OFF (Default Behavior) ---
    print("--- CONFIGURATION 1: Tag Enforcement OFF ---")
    config_off = {"enforce_thinking_answer_tags": False}
    guidance_gen_off = COTGuidanceGenerator(
        total_steps=TOTAL_TRAINING_STEPS, config=config_off
    )

    # Use Scenario 1 target which *has* tags
    messages_sc1 = [
        {
            "role": "user",
            "content": "In the context of Amazon Web Services Cloud, A company has multiple accounts in an organization in AWS Organizations. The company's SecOps team needs to receive an Amazon Simple Notification Service (Amazon SNS) notification if any account in the organization turns off the Block Public Access feature on an Amazon S3 bucket. A DevOps engineer must implement this change without affecting the operation of any AWS accounts. The implementation must ensure that individual member accounts in the organization cannot turn off the notification. Which solution will meet these requirements? Choices: Designate an account to be the delegated Amazon GuardDuty administrator account... Create an AWS CloudFormation template... Turn on AWS Config... Turn on Amazon Inspector... Can you carefully choose the correct answer...",
        }
    ]
    # Generation *without* tags
    last_gen_sc1_no_tags = """
    Okay, let's break this down. The core need is centralized, non-bypassable notification when S3 Block Public Access is turned off.
    GuardDuty is for threats, not config changes directly like this. Inspector is for vulnerabilities. Config can track it, but the Conformance Pack + SSM method seems complex.
    The CloudFormation StackSets approach deploying an EventBridge rule targeting the CloudTrail `s3:PutBucketPublicAccessBlock` event seems most direct. StackSets ensure deployment to all accounts and can prevent modification from member accounts using SCPs or StackSet controls. This directly monitors the API call we care about and notifies via SNS.
    Final Answer: The CloudFormation template solution using StackSets is the best fit.
    """
    # Target *with* tags (from original prompt)
    target_sc1_with_tags = """<thinking>
### Reasoning Structure:

#### Step 1: Review Requirements
1. **Notifications upon disabling Block Public Access on S3 buckets**.
2. **Consistency in notifications that cannot be turned off by individual accounts**.
3. **No disruption to normal operations of AWS accounts**.

#### Step 2: Analyze Choices Against Requirements

1. **GuardDuty Solution:**
   - **Evaluation:** Not a direct fit for the requirement.

2. **CloudFormation Template Solution:**
   - **Evaluation:** Direct and manageable approach that meets requirements.

3. **AWS Config Solution:**
   - **Evaluation:** Strong candidate but slightly more complex.

4. **Amazon Inspector Solution:**
   - **Evaluation:** Not a direct fit.

#### Step 3: Choose the Best Solution
Based on the analysis, the most suitable and direct solution is the **CloudFormation Template Solution**.

### Conclusion:
**Correct Answer:** Create an AWS CloudFormation template... Deploy the stack to every account in the organization by using CloudFormation StackSets.
</thinking>
<answer>
Create an AWS CloudFormation template that creates an SNS topic and subscribes the SecOps team’s email address to the SNS topic. In the template, include an Amazon EventBridge rule that uses an event pattern of CloudTrail activity for `s3:PutBucketPublicAccessBlock` and a target of the SNS topic. Deploy the stack to every account in the organization by using CloudFormation StackSets.
</answer>"""

    print("--- Scenario 1 (Enforcement OFF): Gen without tags, Target with tags ---")
    guidance, loss_report = guidance_gen_off.generate_guidance(
        1, messages_sc1, last_gen_sc1_no_tags, target_sc1_with_tags, domain="general"
    )
    print(
        f"Guidance: {guidance}"
    )  # Should be standard guidance, maybe about structure/detail
    print(
        f"Loss Report: {loss_report}"
    )  # Structure loss should NOT be penalized for missing tags
    print("-" * 20)

    # --- Configuration 2: Tag Enforcement ON ---
    print("\n--- CONFIGURATION 2: Tag Enforcement ON ---")
    config_on = {
        "enforce_thinking_answer_tags": True,
        "loss_parameters": {  # Maybe increase structure weight when enforcing
            "structure_weight": 1.5,
            "content_weight": 1.5,
            "math_weight": 1.5,
            "topic_weight": 0.5,
            "guidance_weight": 0.5,
        },
    }
    guidance_gen_on = COTGuidanceGenerator(
        total_steps=TOTAL_TRAINING_STEPS, config=config_on
    )

    print("--- Scenario 1 (Enforcement ON): Gen without tags, Target with tags ---")
    guidance, loss_report = guidance_gen_on.generate_guidance(
        1, messages_sc1, last_gen_sc1_no_tags, target_sc1_with_tags, domain="general"
    )
    print(f"Guidance: {guidance}")  # EXPECT GUIDANCE ABOUT MISSING <thinking> TAG
    print(
        f"Loss Report: {loss_report}"
    )  # EXPECT HIGH STRUCTURE LOSS due to missing tags
    print("-" * 20)

    # Generation *with* tags but maybe slightly different content / wrong answer
    last_gen_sc1_with_tags_wrong = """<thinking>
    Let's think. We need alerts if S3 public access is disabled. Non-bypassable.
    Option A (GuardDuty): Finds threats, not really config changes. Maybe some finding relates? Seems indirect.
    Option B (CloudFormation): Uses CloudTrail `s3:PutBucketPublicAccessBlock`. Deploys via StackSets. StackSets can enforce central control. This looks promising.
    Option C (Config): Uses Config rules. `s3-bucket-level-public-access-prohibited`. This rule checks if public access is *allowed*, not the specific API call for *turning it off*. Notifications via SSM seem indirect.
    Option D (Inspector): Checks vulnerabilities. Not relevant.
    So, CloudFormation (B) seems best. Wait, Config (C) uses a specific rule `s3-bucket-level-public-access-prohibited`. That sounds very relevant too. Maybe Config is better as it's designed for compliance checks? Let's go with Config.
    </thinking>
    <answer>
    Turn on AWS Config across the organization. In the delegated administrator account, create an SNS topic. Subscribe the SecOps team's email address to the SNS topic. Deploy a conformance pack that uses the s3-bucket-level-public-access-prohibited AWS Config managed rule in each account and uses an AWS Systems Manager document to publish an event to the SNS topic to notify the SecOps team.
    </answer>
    """

    print(
        "--- Scenario 2 (Enforcement ON): Gen with tags (wrong answer), Target with tags ---"
    )
    messages_sc2 = messages_sc1 + [
        {"role": "assistant", "content": last_gen_sc1_no_tags},
        {
            "role": "user",
            "content": "[CoT Guidance: Please structure your response using `<thinking>...</thinking>` tags...]",
        },
    ]  # Simulate previous step
    guidance, loss_report = guidance_gen_on.generate_guidance(
        2,
        messages_sc2,
        last_gen_sc1_with_tags_wrong,
        target_sc1_with_tags,
        domain="general",
    )
    print(f"Guidance: {guidance}")  # Should be about content/answer error now.
    print(
        f"Loss Report: {loss_report}"
    )  # EXPECT LOW STRUCTURE LOSS (tags present), HIGH CONTENT LOSS (wrong answer)
    print("-" * 20)

    # Generation *with* tags and *correct* answer
    last_gen_sc1_with_tags_correct = """<thinking>
    Okay, the goal is a central, tamper-proof alert when S3 Block Public Access is disabled on *any* account bucket.
    1. GuardDuty: Focuses on malicious activity/threats, not specific configuration API calls like PutBucketPublicAccessBlock. Less direct.
    2. CloudFormation + StackSets: Can deploy an EventBridge rule listening for the `s3:PutBucketPublicAccessBlock` CloudTrail event. StackSets allow central deployment and management, which helps meet the "cannot be turned off" requirement (especially if paired with SCPs). This directly targets the event.
    3. AWS Config: Can use the `s3-bucket-level-public-access-prohibited` rule. This checks the *state* (is public access allowed?), not the *action* of disabling the block. While related, it might not trigger exactly on the API call itself. Using SSM for notifications adds complexity.
    4. Inspector: Focuses on workload vulnerabilities (EC2, ECR, Lambda), not S3 config changes. Irrelevant.
    Comparing B and C: CloudFormation targeting the CloudTrail event seems more precise for the trigger condition (disabling the block) and StackSets provide a strong mechanism for central enforcement.
    Therefore, the CloudFormation approach is the most direct and robust solution.
    </thinking>
    <answer>
    Create an AWS CloudFormation template that creates an SNS topic and subscribes the SecOps team’s email address to the SNS topic. In the template, include an Amazon EventBridge rule that uses an event pattern of CloudTrail activity for `s3:PutBucketPublicAccessBlock` and a target of the SNS topic. Deploy the stack to every account in the organization by using CloudFormation StackSets.
    </answer>
    """

    print(
        "--- Scenario 3 (Enforcement ON): Gen with tags (correct answer), Target with tags ---"
    )
    messages_sc3 = messages_sc2 + [
        {"role": "assistant", "content": last_gen_sc1_with_tags_wrong},
        {
            "role": "user",
            "content": "[CoT Guidance: Content/answer seems incorrect...]",
        },
    ]  # Simulate previous step
    guidance, loss_report = guidance_gen_on.generate_guidance(
        3,
        messages_sc3,
        last_gen_sc1_with_tags_correct,
        target_sc1_with_tags,
        domain="general",
    )
    print(f"Guidance: {guidance}")  # Should be None or generic/verification guidance.
    print(
        f"Loss Report: {loss_report}"
    )  # EXPECT LOW LOSS across structure and content.
    print("-" * 20)

    print(
        "--- Scenario 4 (Enforcement ON): Math Problem - Correct Tags, Wrong Answer ---"
    )
    # Re-use Finance problem from original example
    messages_sc4 = [
        {
            "role": "user",
            "content": "Calculate simple interest on $500 at 5% annual rate for 3 years.",
        }
    ]
    last_gen_sc4_tags_wrong = "<thinking>Formula is I = P*R*T. P=500, R=5, T=3. I = 500 * 5 * 3. 500*5=2500. 2500*3=7500.</thinking><answer>The simple interest is $7500.</answer>"  # Forgot rate conversion
    target_sc4_tags = """<thinking>
1.  **Identify Variables:** Principal (P) = $500, Rate (R) = 5% per year = 0.05, Time (T) = 3 years.
2.  **Formula:** Simple Interest (I) = P * R * T.
3.  **Substitute:** I = $500 * 0.05 * 3 years.
4.  **Calculate:** I = $25 * 3 = $75.
</thinking>
<answer>The simple interest earned is $75.</answer>"""
    guidance, loss_report = guidance_gen_on.generate_guidance(
        1, messages_sc4, last_gen_sc4_tags_wrong, target_sc4_tags, domain="finance"
    )  # Domain might be inferred as math/finance
    print(f"Guidance: {guidance}")  # Expect guidance about math error / content error.
    print(
        f"Loss Report: {loss_report}"
    )  # Expect low structure loss, high content/math loss.
    print("-" * 20)
