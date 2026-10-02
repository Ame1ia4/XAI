**Title:** Which LIME should I trust? Concepts,
Challenges, and Solutions

**Link:** https://arxiv.org/pdf/2503.24365

**Key Takeaways:** This paper addresses the different adaptations of LIME and provides a holistic overview of advancements in LIME.

**Abstract:** As neural networks become dominant in essential systems,
Explainable Artificial Intelligence (XAI) plays a crucial role in fostering trust and detecting potential misbehavior of opaque models. LIME
(Local Interpretable Model-agnostic Explanations) is among the most
prominent model-agnostic approaches, generating explanations by approximating the behavior of black-box models around specific instances.
Despite its popularity, LIME faces challenges related to fidelity, stability, and applicability to domain-specific problems. Numerous adaptations
and enhancements have been proposed to address these issues, but the
growing number of developments can be overwhelming, complicating efforts to navigate LIME-related research. To the best of our knowledge,
this is the first survey to comprehensively explore and collect LIME’s
foundational concepts and known limitations. We categorize and compare its various enhancements, offering a structured taxonomy based
on intermediate steps and key issues. Our analysis provides a holistic
overview of advancements in LIME, guiding future research and helping practitioners identify suitable approaches. Additionally, we provide a
continuously updated interactive website, Which LIME Should I Trust?,
offering a concise and accessible overview of the survey.

**AI Summary:** . "Which LIME should I trust? Concepts, Challenges, and Solutions" (arXiv 2503.24365)

*Summary:* This is a survey of LIME and its variants, built from a review of over 20,000 papers. It organises LIME extensions along two axes: the technical modification made within the LIME pipeline, and the specific issue each one addresses. The pipeline has four steps: feature generation, sample generation, feature attribution (the surrogate fit), and explanation representation. The issues are locality, fidelity, interpretability, stability, and efficiency. It maps each variant to the step it changes and to the modality it targets. It also criticises the field's practices. About half of the methods lack code, and many papers compare only against vanilla LIME. 

*Relevance to your topic:* high, and it is the best starting point.

- A ready-made map of the tabular literature. Its Table 2 lists dozens of tabular-specific variants (e.g. ILIME, Kernel-LIME, K-LIME, LIME-SUP, GMM-LIME, BMB-LIME, US-LIME, LIMEtree, UnRAvEL-LIME, LSLIME, QLIME, bLIMEy). These are candidates for the "quality" side of your comparison.
- The cost/quality tension is stated explicitly. The survey notes that the issues are interdependent: increasing locality can hurt efficiency, while lowering efficiency can help stability. That is essentially your research question. 
arxiv
- Where the cost comes from. It attributes cost to perturbation generation, black-box predictions, and surrogate fitting. In the notation, the perturbation mask is a binary vector in {0,1}^d, where d is the feature count, which is the quantity that grows in your setting.
- Scalability is a named evaluation property (Table 3, alongside Efficiency and Consistency), but the survey does not benchmark it. It also says a standard LIME evaluation framework is missing, so you would be building your own protocol.

*Gap:* The survey does not analyse how any variant behaves as dimensionality grows. That question is still open, which makes it a reasonable motivation for your work.

**Paper's Value:** This paper poses more relevance as it discusses the computational inefficieny's  of LIME and limitations in the handling of certain types of data as one of five issue categories. The paper mentions that evaluation practice is inconsistent and about half the methods lack code, a possible reason why scaling comparisons are hard to find.
