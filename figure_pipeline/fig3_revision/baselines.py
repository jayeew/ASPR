"""Independent adapted-baseline scientific step ordering."""
STEPS = {
    'eacl': [
        ('structured_extraction', 'Independently extract contributions, citation contexts, scientific problem, methods, results and limitations from manuscript.'),
        ('research_landscape', 'Compare extracted advances to retrieved historical work; distinguish closest prior methods and unresolved gaps.'),
        ('novelty_assessment', 'Assess incremental versus distinctive elements, necessary qualifications and supported historical differences.'),
        ('review_guidance', 'Provide evidence-grounded guidance about innovation strengths, doubts and what cannot be established.')],
    'reviewgrounder': [
        ('paper_review', 'Draft a manuscript-grounded critical review of contributions, methods, results and limitations.'),
        ('related_work', 'Compare draft claims with retrieved literature; identify genuine comparisons and evidence gaps.'),
        ('results_analysis', 'Analyze experimental results and baselines from the manuscript. Preserve unsupported or missing comparisons.'),
        ('insight_mining', 'Identify substantive scientific insights, cross-work relationships and justified limitations from preceding analyses.'),
        ('review_refinement', 'Refine using rubric: contribution fidelity, historical difference, evidence adequacy, scope and uncertainty, whole-paper insight. Correct unsupported claims without inventing findings.')],
}

