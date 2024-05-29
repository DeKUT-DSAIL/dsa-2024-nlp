# DSA 2024 NLP

© DSA 2024. Apache License 2.0.

Contributors : `Samuel Mbatia`, `Teofilo Ligawa`, `Cedric Kiplimo`, `Andreas Damianou`


**Introduction**

[Natural language processing (NLP)](https://www.oracle.com/ke/artificial-intelligence/what-is-natural-language-processing/) is the ability of a computer program to understand human language as it's spoken and written -- referred to as natural language. It's a component of artificial intelligence (AI).Natural language processing has the ability to interrogate the data with natural language text or voice.


## 1. Prelab Work
In preparation for the main NLP lab session at DSA 2024, we have prepared some prelab work to help you warm up. Just like the main lab, the prelab is also broken down per topic into the following sections:
* Basic NLP
* Large Language Models (LLMs) with Google Colab and Hugging Face Models
* Prompt Engineering

A separate notebook has been provided for each of these topics and you can run them on Colab. **We strongly advise** that you spare some time to complete these exercises before the main lab session.
<!-- #region -->
### A. Basic NLP
For instructions on how to complete each exercise, refer to the [prelab folder](./pre-lab/README.md).
In the basic NLP section, the focus shall be to familiarize with some foundational concepts in NLP.

Some of the areas covered in this section include:
* Stemming
* Lemmatization
* Stopwords
* Tokenization
* Text Vectorization
* Next word probability

#### Learning Objectives
* Distinguish between stemming and lemmatization.
* Describe stopwords and why they are removed.
* Describe tokenization in the context of NLP.
* Understand various vectorization techniques.
* Use probability to predict the next word in a sentence.

To complete the tasks in this part, <a target="_blank" href="https://colab.research.google.com/github/DeKUT-DSAIL/DSA-2024-NLP/blob/main/pre-lab/simple_nlp_prelab.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

If you wish to save changes you make to the notebook, please create your own copy.

<!-- #region -->
### B. Large Language Models (LLMs) with Google Colab and Huggingface
A large language model (LLM) is a statistical language model, trained on a massive amount of data, that can be used to generate and translate text and other content, and perform other natural language processing (NLP) tasks.
LLMs are typically based on deep learning architectures, such as the Transformer developed by Google in 2017, and can be trained on billions of text and other content.[[2]](https://cloud.google.com/ai/llms)
Text-driven LLMs are used for a variety of natural language processing tasks, including text generation, machine translation, text summarization, question answering, and creating chatbots that can hold conversations with humans.

LLMs can also be trained on other types of data, including code, images, audio, video, and more. Google’s [Codey](https://cloud.google.com/vertex-ai/docs/generative-ai/code/code-models-overview), [Imagen](https://cloud.google.com/vertex-ai/docs/generative-ai/image/overview) and [Chirp](https://cloud.google.com/vertex-ai/docs/generative-ai/speech/speech-to-text) are examples of such models that will spawn new applications and help create solutions to the world’s most challenging problems.

LLMs are pre-trained on a massive amount of data. They are extremely flexible because they can be trained to perform a variety of tasks, such as text generation, summarization, and  Question Answering e.t.c. They are also scalable because they can be fine-tuned to specific tasks, which can improve their performance.

Here we shall utilize [huggingface](https://huggingface.co/models)[[5]](https://huggingface.co/docs/hub/index) pretrained models to accomplish tasks such as Text generation,Text summmarization and Question answering. Using the steps above to create a huggingface account and accessing your secret token which will eneble the usage of models in colab , make sure the token is inputted into your google colab as shown in the steps above.

To complete the tasks in this part, click <a target="_blank" href="https://colab.research.google.com/github/DeKUT-DSAIL/DSA-2024-NLP/blob/main/pre-lab/DSA_LLM_PreLab.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> to open the notebook.

If you wish to save changes you make to the notebook, please create your own copy.
#### Learning Objectives
What will be covered in the notebook:
* Use a pretrained models in huggingface to accomplish Text summarization
* Use a pretrained models in huggingface to accomplish Text generation
* To do exercises to challenge yourself

<!-- #endregion -->

## 2. Main Lab
If you have not attempted the [prelab exercises](#prelab-work), we encourage you to have a look at them first. 

**Topics:**

Content: `Natural Language Processing`,

Level: `Beginner`

**Learning Objectives:**
- Introduce you to Natural language processing tasks with python and machine learning.

**Prerequisites:**
- Basic knowledge of [Python Programming](https://ocw.mit.edu/courses/6-0001-introduction-to-computer-science-and-programming-in-python-fall-2016/)
- Prior interaction with large language models e.g. by using ChatGPT, will be helpful.
- A [Google Colab](https://colab.research.google.com/) account
- A [Hugging Face](https://huggingface.co/join)  account

- When done creating the hugging face account ,you have to create an acess token and load it into your Google Colab so as to acess some models.
- 
  * access the settings part on the top right hand side
  ![HuggingFace Setting](./assets/access.png)
  
  * access the tokens and create one.Choose the write type.
  ![Huggingface Access Tokens](./assets/tokens.png)
  
  * input it into your colab,then you are ready to go.
  ![Secret Key in Colab](./assets/secretkey.png)

<!-- #region -->
### A. Basic NLP
In the basic NLP section, the focus shall be to familiarize with some foundational concepts in NLP.

Some of the areas covered in this section include:
* Word Embeddings
* Named Entity Recognition
* Sentence Segmentation
* Parts of Speech Tagging

#### Learning Objectives
* Understand sentence segmentation.
* Understand various ways of getting word embeddings.
* Understand Part of Speech (POS) tagging.
* Understand Named Entity Recognition (NER).

To complete the tasks in this part, <a target="_blank" href="https://colab.research.google.com/github/DeKUT-DSAIL/DSA-2024-NLP/blob/main/main-lab/basic_nlp_lab.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

If you wish to save changes you make to the notebook, please create your own copy.

<!-- #endregion -->

<!-- #region -->
### B. Large Language Models (LLMs) with Google Colab and Huggingface
A large language model (LLM) is a statistical language model, trained on a massive amount of data, that can be used to generate and translate text and other content, and perform other natural language processing (NLP) tasks.
LLMs are typically based on deep learning architectures, such as the Transformer developed by Google in 2017, and can be trained on billions of text and other content.[[2]](https://cloud.google.com/ai/llms)
Text-driven LLMs are used for a variety of natural language processing tasks, including text generation, machine translation, text summarization, question answering, and creating chatbots that can hold conversations with humans.

LLMs can also be trained on other types of data, including code, images, audio, video, and more. Google’s [Codey](https://cloud.google.com/vertex-ai/docs/generative-ai/code/code-models-overview), [Imagen](https://cloud.google.com/vertex-ai/docs/generative-ai/image/overview) and [Chirp](https://cloud.google.com/vertex-ai/docs/generative-ai/speech/speech-to-text) are examples of such models that will spawn new applications and help create solutions to the world’s most challenging problems.

LLMs are pre-trained on a massive amount of data. They are extremely flexible because they can be trained to perform a variety of tasks, such as text generation, summarization, and  Question Answering e.t.c. They are also scalable because they can be fine-tuned to specific tasks, which can improve their performance.

Here we shall utilize [huggingface](https://huggingface.co/models)[[5]](https://huggingface.co/docs/hub/index) pretrained models to accomplish tasks such as Text generation,Text summmarization and Question answering. Using the steps above to create a huggingface account and accessing your secret token which will eneble the usage of models in colab , make sure the token is inputted into your google colab as shown in the steps above.

To complete the tasks in this part, click Open the [LLMs Lab Notebook](https://github/DeKUT-DSAIL/DSA-2024-NLP/blob/main/main-lab/DSA_Lab_LLMs.ipynb) <a target="_blank" href="https://colab.research.google.com/github/DeKUT-DSAIL/DSA-2024-NLP/blob/main/main-lab/DSA_Lab_LLMs.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> to open the notebook.

If you wish to save changes you make to the notebook, please create your own copy.

#### Learning Objectives
What will be covered in the notebook:
* Use a pretrained models in huggingface to accomplish Text summarization
* Use a pretrained models in huggingface to accomplish Text generation
* Use a pretrained model in huggingface together with the haystack framework to accomplish Question answering
* To do exercise to challenge yourself


### C. Prompt Engineering with LLaMA-2
**Duration: 30 minutes**

Prompt engineering is the discipline of developing and optimising prompts to effectively use large language models (LLMs) to achieve desired outputs for a wide variety of applications, including research. By developing prompt engineering skills, we are enabled to better understand the capabilities and limitations of these LLMs.

The key aspects of prompt engineering include, but are not limited to:
* Crafting clear prompts: The model's output is significantly affected by the model's output. To get accurate and relevant responses, prompts should be clear, concise, and specific.
* Providing context: Prompts that include sufficient context within them help the models understand the background and generate more informed responses. Contexts can involve giving background information, setting the scene, or specifying the desired format of the answer.
* Iterative refinement: Prompt engineering is often an iterative process where initial prompts are continuously adjusted and refined to improve the quality of the response.
* Instruction precision: Explicity stating what you want from the model can dramatically improve outcomes. Using words like "list", "describe", etc. help guide the model more effectively.
* Balancing length and detail: Although detailed prompts can provide more guidance, overly long prompts tend to confuse the model. Striking a balance between providing enough details and maintaining brevity is important.
* Leveraging special tokens: Some models allow the use of special tokens or specific structures to control responses, such as separators or format indicators. `LLaMA-2` is one such model.

#### Prerequisites
This lab is targeted at Python developers who have some familiarity with LLMs, such as by using ChatGPT or Gemini, but have limited experience in working with LLMs in a programmatic way.

If you're familiar with the underpinnings of LLMs, you'll have a slight advantage. However, familiarity with basic Python and a basic understanding of LLMs will be sufficient to help you get a lot out of this course.

For this lab, we shall be working with the `LLaMA-2` model available at [HuggingFace](https://huggingface.co/TheBloke/Llama-2-13B-chat-GPTQ). In order to download this model for use in this notebook, you will need to install the [transformers](https://pypi.org/project/transformers/) package. Not to worry, the steps for installing it are baked into this Colab notebook and you will not need to take any extra steps, except if you choose to run the notebook locally.


#### Learning Objectives
1. Use a `transformers` pipeline to generate responses from a LLaMA-2 LLM.
2. Iteratively write precise prompts to get the desired output from the LLM.
3. Work with the LLaMA-2 prompt template to perform instruction fine-tuning.
4. Use LLaMA-2 to generate JSON data for potential use in downstream processing tasks

To complete the tasks in this part, click <a target="_blank" href="https://colab.research.google.com/github/DeKUT-DSAIL/DSA-2024-NLP/blob/main/pre-lab/prompt_engineering_prelab.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> to open the notebook.

If you wish to save changes you make to the notebook, please create your own copy.


<!-- #endregion -->

### References
1. Oracle. (n.d.). *What is Natural Language Processing?* Retrieved from [https://www.oracle.com/ke/artificial-intelligence/what-is-natural-language-processing/](https://www.oracle.com/ke/artificial-intelligence/what-is-natural-language-processing/)
2. Google Cloud LLMs[https://cloud.google.com/ai/llms](https://cloud.google.com/ai/llms)
3. McEnery, T., Xiao, R., & Tono, Y. (2006). *Corpus-based language studies: An advanced resource book*. Taylor & Francis.
4. Wikipedia. (n.d.). *Pointwise mutual information*. Retrieved from [https://en.wikipedia.org/wiki/Pointwise_mutual_information](https://en.wikipedia.org/wiki/Pointwise_mutual_information)
5. Hugging face Documentation[https://huggingface.co/docs/hub/index](https://huggingface.co/docs/hub/index)


**Contributors**

| Name              | GitHub                                            | Affiliation                                     |
|-------------------|---------------------------------------------------|-------------------------------------------------|
| Cedric Kiplimo    | [@kiplimock](https://github.com/kiplimock)         | [DeKUT-DSAIL](https://dekut-dsail.github.io)    |
| Teofilo Ligawa    | [@teofizzy](https://github.com/teofizzy)          | [DeKUT-DSAIL](https://dekut-dsail.github.io)    |
| Samuel Mbatia     | [@mbatiasonic](https://github.com/mbatiasonic)    | [DeKUT-DSAIL](https://dekut-dsail.github.io)    |
| Andreas Damianou  | [@adamian](https://github.com/adamian)            | [Spotify](http://andreasdamianou.com/)          |


## Questions

Please ask any questions through this [form](https://forms.gle/dbd19Sk1VPydsAtp8)
