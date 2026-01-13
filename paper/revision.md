2. IDKT Model

¿Porque Transformers? Antes no se menciona (en la intro)

Datasets

No se justifica porque esos datasets, de donde salen

Al hablar de las metricas: no queda claro cual es mi aportacion y que es lo que he utilizado de lo existente. 


4.1 Predictive Performance

NO se sabe porque esos modelos, de donde salen. 

.1. Introduction

Poner un textointroductorio, antes de subsecciones mencionado que se pone sobre la mesa, que es rep. grounding, semantic interpretability, necessity of individualization...

Adelantar los conceptos (importantes) de los que se va a hablar. 


1.2 Semantic Interpretability

No se sabe muy bien porque se empieza a hablar de esto de repente, dado que aparece en el tutlo ....

empezar la into planteando todos los conceptos con los que se vaa trabajar y luego esos conceptos se vam mencionado en el estado del arte. 


Doble embudo: centrar, abrir

## Abstract

"We introduce a formal validation framework" ...

Si es novedoso -> hay que habalr de trabajo relacionado, que han hecho otros. 

No habalr de framework, un revisor lo puede rechazar ...

Hay que mostar que otra metrica hay ,y si no las hay decirlo. 


Falta una tabla donde se comparen los diferentes enfoques analizados incluyendo una ultima fila para iDKT. Columnas con aspectos de interes de la propuesta subrayando los puntos que yo aporto. Mostrara metodos o algorimos cubietos por otros y como iDKT los cubre toos. Incluir metricas, limitaciones de otras aproximaciones. Que se vea claramente lo que aportamos respecto al resto. Serviria para aclarar y poner en valor. 

2.1 Positioning Overview

Aqui podriamos apoyarnos en esa tabla. 

Rehacer diagrama Von Rueden. 

"The integration of domain knowledge into deep learning has been broadly explored" vuelve al estado del arte. Separarlo...

La tabla que comenta Olga puede basarse en las taxonomia de xAI o de IML. Mencionar los trabajos que hay, lo que aportan y lo que no. 

- Indvidualization 

No se explica bien, en el abstract no se explica bien. Habria que explicarlo en la Intro. 

- BKT

Revisar acronimos


- Faltan referencias (pagina 2)

- "While providing insights into controlled scenarios", no se entiende a que se refiere lo de "controlled", si se quere remarcar que nuestra propuesta puede ser aplicada a otros escenarios meterlo en la intro y luego al final se retoma que iDKT se puede usar en otros escenarios. 

- 1.2 "To achieve practical interpretability without compromising general applicability..." no queda claro, ponerlo mas explicito 

- "3 criterios de S. Interpretabilidad". ¿De donde han salido? Si no es estado del arte, tiene que explicarse en mi propuesta 

1.2. Semantic Interpretability 

linea 118: estamos metiendo logros en lo que deberia ser un SOTA. 

Justificar el uso de los datasets... por qué se han escogido. Quizás se podria hacer otra tabla con esto. 

Los modelos de la comparativa: porque se han escogido, que cubren y que no. Que aporta iDKT. 


# Arquitectura Transformers

Que se aporta y que es estandar. 

Destacar trabajos que utilizan transformers y que aportan ellos, comparar idkt con ellos. 

## "In standard educational datasets, such as ASSISTments 2009, ASSISTments 2015, 222
Algebra 2005, and others [30]" al SOTA


3. Experimental Setup

quitar "The experimental validation of iDKT is guided by the following research questions:"

Parrafo explicando 

3.1. Implementation Details

Los valores de los parametros con los que se entrenan los modelos, ¿de donde han salido?. Justificarlos. 

1.4. Research Hypotheses 

Metricas de Interpretabilidad: cuales, porque, comparar con el SOTA. Si son nuevas Validarlas 


Merticas: queda por justificar porque es una metrica adecuada ... Quizas en vez de hablar un framework de validacion, simplemente presentarlas como un medio de validacion, no como una aportacion en si misma. 


- Referencias: revisarlas (en algunas faltan datos, 7. 10.). Uso demasiados preprints (que  no estan refreenciados por pares, referenciar siempre la version publicada)

- Plantillas. Quitar:  project administration, O. C. S.; funding acquisition, O. C. S

- Revisar revisores externos de la tesis (Nercesario antes depositar). Olga está buscando a los revisores externos. Incluir referencias a Cristobal Romero (que ha mostrado interés en revisarla), Estefania Barroso, David, Roldan (JC). Univ Valencia: Miguel Arevalillo. Yuyan Wu, Pablo Arnau-Gonzalez

- Graficas. No son esenciales (no te van a rechazar por ellas, sí por no seguir un ametodología adecuada o por falta de consistencia), se pueden mejorar despues. Es mucho más critico todo lo anterior. 

- En la Intro justificar porqué luego se presentan las gráficas del final (apartados de contextualización, individualización, etc.)


