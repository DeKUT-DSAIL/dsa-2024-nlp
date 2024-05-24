# DSA 2024 NLP

© DSA2024. Apache License 2.0.

Contributors : `Samuel Mbatia`, `Teofilo Ligawa`, `Cedric Kiplimo`, `Andreas Damianou`


**Introduction**

Natural language processing (NLP) is the ability of a computer program to understand human language as it's spoken and written -- referred to as natural language[[1]](https://www.oracle.com/ke/artificial-intelligence/what-is-natural-language-processing/). It's a component of artificial intelligence (AI).Natural language processing has the ability to interrogate the data with natural language text or voice.

**Topics:**

Content: `Natural Language Processing`,

Level: `Beginner`

**Learning Objectives:**
- Introduce you to Natural language processing tasks with python and machine learning.

**Prerequisites:**
- Basic knowledge of [Python Programming](https://ocw.mit.edu/courses/6-0001-introduction-to-computer-science-and-programming-in-python-fall-2016/)
- A [Google Colab](https://colab.research.google.com/) account
- A [Hugging Face](https://huggingface.co/join)  account

<!-- #region -->
### A. Basic NLP
In the basic NLP section, the focus shall be to familiarize with some foundational concepts in NLP.

Some of the areas covered in this section include:
* Stemming
* Lemmatization
* Stopwords
* Tokenization
* Text Vectorization
* Next word probability

#### Objectives
* Distinguish between stemming and lemmatization.
* Describe stopwords and why they are removed.
* Describe tokenization in the context of NLP.
* Understand various vectorization techniques.
* Use probability to predict the next word in a sentence.

* Open the [Simple NLP Pre-Lab Notebook]() <a target="_blank" href=""><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> and follow the steps as outlined in the notebook. and follow the instructions in the noteook.

<!-- #endregion -->

<!-- #region -->
### B. Large Language Models (LLMs) with Google Colab and Huggingface
A large language model (LLM) is a statistical language model, trained on a massive amount of data, that can be used to generate and translate text and other content, and perform other natural language processing (NLP) tasks.
LLMs are typically based on deep learning architectures, such as the Transformer developed by Google in 2017, and can be trained on billions of text and other content.[[2]](https://cloud.google.com/ai/llms)
Text-driven LLMs are used for a variety of natural language processing tasks, including text generation, machine translation, text summarization, question answering, and creating chatbots that can hold conversations with humans.

LLMs can also be trained on other types of data, including code, images, audio, video, and more. Google’s [Codey](https://cloud.google.com/vertex-ai/docs/generative-ai/code/code-models-overview), [Imagen](https://cloud.google.com/vertex-ai/docs/generative-ai/image/overview) and [Chirp](https://cloud.google.com/vertex-ai/docs/generative-ai/speech/speech-to-text) are examples of such models that will spawn new applications and help create solutions to the world’s most challenging problems.

LLMs are pre-trained on a massive amount of data. They are extremely flexible because they can be trained to perform a variety of tasks, such as text generation, summarization, and  Question Answering e.t.c. They are also scalable because they can be fine-tuned to specific tasks, which can improve their performance.

Here we shall utilize [huggingface](https://huggingface.co/models) pretrained models to accomplish tasks such as Text generation,Text summmarization and Question answering. Using the steps above to create a huggingface account and accessing your secret token which will eneble the usage of models in colab , make sure the token is inputted into your google colab as shown in the steps above.

Open the [LLMs Notebook]() and follow the instructions in the noteook.

What will be covered in the noteook:

(1) Use a pretrained models in huggingface to accomplish Text summarization

(2) Use a pretrained models in huggingface to accomplish Text generation

(3) Use a pretrained models in huggingface to accomplish Question Answering

(4) To do exercises to challenge yourself


<!-- #endregion -->

**References**

1. [HuggingFace CV examples](https://huggingface.co/datasets)
3. Oracle. (n.d.). *What is Natural Language Processing?* Retrieved from [https://www.oracle.com/ke/artificial-intelligence/what-is-natural-language-processing/](https://www.oracle.com/ke/artificial-intelligence/what-is-natural-language-processing/)
4. McEnery, T., Xiao, R., & Tono, Y. (2006). *Corpus-based language studies: An advanced resource book*. Taylor & Francis.
5. Wikipedia. (n.d.). *Pointwise mutual information*. Retrieved from [https://en.wikipedia.org/wiki/Pointwise_mutual_information](https://en.wikipedia.org/wiki/Pointwise_mutual_information)


**Inspiration**



| Name              | GitHub                                            | Affiliation                                     |
|-------------------|---------------------------------------------------|-------------------------------------------------|
| Cedric Kiplimo    | [@kiplimok](https://github.com/kiplimock)         | [DeKUT-DSAIL](https://dekut-dsail.github.io)    |
| Teofilo Ligawa    | [@teofizzy](https://github.com/teofizzy)          | [DeKUT-DSAIL](https://dekut-dsail.github.io)    |
| Samuel Mbatia     | [@mbatiasonic](https://github.com/mbatiasonic)    | [DeKUT-DSAIL](https://dekut-dsail.github.io)    |
| Andreas Damianou  | [@adamian](https://github.com/adamian)            | [Spotify](http://andreasdamianou.com/)          |


## Questions

Please ask any questions through this [form](https://forms.gle/cWTba8SHamqhtrP38)
