Referencias: 

- Meter paper nuestro
- Referenciar paper Pardos. En los otros dos ver porque nos referencian y darle la vuelta ... en "Discussion" se podría retomar lo que digan estos papers y que nosotras no hemos contemplado. 

Plazo: pedir hasta el 17. 

# Reviewers Comments

Empezar agradeciendo tiempo, lectura cuidadosa, lo mucho que aportan, lo que hay de valor en los comentarios. 

## Reviewer 4

Comments and Suggestions for Authors: 

To train the model, datasets were used that were mainly mathematically oriented - 4 out of the 5 datasets were task results. The fifth dataset was not specified. This means that the evaluation was carried out mainly on one type of educational data. Therefore, either the title should be narrowed down and specified that it is about training in the field of mathematics. Or the study should be expanded to include other different datasets.

### R: justificar porque se escogen estos datasets (e.g. son los que se eferencian en el benchmark pykt...); en "Limitaciones" indicar: que los datasets ... matematicas ... seria necesario explorar otro tipo datasets; el analisis es independiente del dominio; al revisor explicar esto y que se ha incluido en las "limitaciones".  

Since gTransformer is based on the concepts of BKT, it also takes on some of its shortcomings - how does gTransformer capture the complex, nonlinear dependencies in student behaviour?

### R: Posibilidades: 1) explicarlo a nivel técnico (cómo lo hace); 2) cómo sabemos que funciona: Figuras 7 y plots; 3) "gracias ... se ha aprovechado para clarificar" ... ver si hay que explicarlo mejor en el paper

For the purposes of the research, only two of the four parameters of Bayesian Knowledge Tracing (BKT) – initial knowledge (P(L_0)) and learning rate (P(T)) – undergo the grounding process. The parameters Guess and Slip are used as fixed values ​​from the population, which limits full individualisation.
Please extend the study by analysing how the model will respond if non-zero values ​​are set for Guess and Slip.

### R: buscar referencias donde se hable de los guess and slip y porque esta justificado tomarlos a nivel de poblacion; "lo que dices es cierto ... pero consideramos que tiene sentido hacerlo así porque ... simplificar la arquitectura por ... tiene sentido a nivel de poblacion. Ponerlo en "Limitaciones" (o en la "Discusion"): es una primera aproximacion ... seria interesante incluirlos ... pero esta fura del alcance inicial en el que nos hemos centrado cuyo objetivo es una primera validacion inicial ... en trabajos futuros tiene sentido extender el analisis a los 4 parametros por ... aunque esto supone crear una nueva version de la arquitectura, lanzar un training costoso en terminos computacionales". Ver si se puede poner tambien en la "Discusion". 

Comments on the Quality of English Language: 

Please, check spelling. Please, use UK English, not US. 

### R: revisar palabras típicas que difieren entre inglés británico y americano: modeling, stadandardisation, etc. 

## Reviewer 1

This paper introduces gTransformer, a novel grounded Transformer model that bridges deep learning performance with intrinsic interpretability through Representational Grounding. The work addresses an important problem, and I have several suggestions to strengthen the manuscript:

1. Introduction Section Needs Condensing
The introduction section is currently too lengthy. Specifically, the first five paragraphs should be more concise to clearly pinpoint the research gap. Much of the detailed discussion of previous research in these opening paragraphs would be more appropriately placed in the Related Work section. Streamlining this section would help readers quickly grasp the paper's core contribution.

### R: tratar de metr subapartados en la Intro para facilitar la lectura. 

2. Methodology Details in Lines 133-151
The content in lines 133-151 should be summarized more concisely. Detailed methodological descriptions should be reserved for the Methodology section rather than appearing in the introduction or results.

3. Tense Consistency in Section 5
Section 5 should consistently use the present tense rather than future tense. For example, line 545 describes current behavior but uses future-oriented language. Additionally, avoid including methodology subtitles within the Results section (see line 571). The Results section should present findings, not restate methodological procedures.

4. Missing Limitations Section
The manuscript would benefit from a dedicated Limitations section discussing the constraints, assumptions, and potential weaknesses of the proposed approach. This transparency strengthens scholarly rigor.

### R: meter seccion "Limitaciones". ¿Se podrian separar Results y Discussion (dentro de Discussion meter dos subapartados: Limitaciones y Trabajos Futuros); metiendo Limitaciones en Discussion se evita que Limitacionbes quede demasiado corto. 

5. Proofreading for Readability
A thorough proofread is recommended to enhance overall readability and flow. Attention to sentence structure, transitions between sections, and consistent terminology would improve the manuscript's clarity.  

### R: meter subapartados en Intro, "se han estructurado mejor las secciones", meter parrafos de enlace del discurso. "sentence structure": frases mas sencillos, "consistent terminology": evitar sinonimos (los inglese prefieren usar siempre los mismos terminos, no usar sinonimos). Usar bullets, subtitulos y divisiones. 

The core contribution of this paper is valuable, and addressing the above organizational and presentational issues will significantly strengthen the paper's impact.

## Reviewer 2

Comments and Suggestions for Authors
This paper presents gTransformer, a deep learning model for interpretable knowledge tracing through representation grounding. The goal is to provide a practical solution achieving both high prediction accuracy and pedagogical interpretability. The study focused on three research questions: (1) tradeoff between performance and interpretability; (2) semantic alignment with theoretical priors; and (3) student-centered personalization. The authors evaluated the proposed model using five benchmark datasets and conducted a comparative analysis of gTransformer with classical knowledge tracing models. Experimental results show that gTransfomrer is competitive regarding prediction accuracy and effective in semantic alignment with theoretical priors and context-aware personalization. 

Overall, the paper is well-structured. The authors gave a clear description of the proposed model and the experimental settings. No critical technical issues were spotted. The research fits the scope of the Applied Sciences journal. The references are appropriate and up to date. The topic should be interesting to certain readers.

I did not spot critical technical issues. However, while introducing the model in Sections 3.1 and 3.2, the authors should provide a detailed description of the layer number, the neuron size of each layer, the function (e.g., drop out, pooling) of each layer, and the key parameters used (e.g., learning rate, optimizer, etc.). 

### R: dar los detalles. Dice que la estructura esta bien, hay que justificar entonces el hecho de que se hagan cambios en la misma ("teniendo en cuenta los comentarios de otros revisores hemos hecho estos cambios ...")

Minor issues exist in the paper (e.g., "i.e." should be "i.e.," and "e.g." should be "e.g.,"). Proofreading should be done before re-submission. 

### R: comma and distinguir i.e and e.g. 

## Reviewer 3

Comments and Suggestions for Authors
This paper demonstrates a clearly strong scientific component, methodologically well-designed to confirm or refute social concepts and learning contexts involving personalized segments with the highest possible degree of statistical reliability. Such elements are rarely measured with comparable rigor in empirically based studies of a similar orientation. The methodology is highly commendable but however a minor suggestion for a slight improvement applies to the discussion section toward the end of the paper and the conclusion. It would be beneficial to elaborate more extensively on the practical implications of the findings, placing greater emphasis on the social dimension and on the broader context of application derived from the results.

### R: en la "Discussion" ... meter titulo de "Practical Applications" ... si no encaja porque ya se está explicando en otras secciones, entonces explicar dónde se estan explicando las aplicaciones prácticas (remarcarlo con subtitulo si se puede) ... "gracias ..."



