const quizData = [
    {
        question: "A company makes forecasts each quarter to decide how to optimize operations to meet expected demand. The company uses ML models to make these forecasts. An AI practitioner is writing a report about the trained ML models to provide transparency and explainability to company stakeholders. What should the AI practitioner include in the report to meet the transparency and explainability requirements?",
        options: [
            "A. Code for model training",
            "B. Partial dependence plots (PDPs)",
            "C. Sample data for training",
            "D. Model convergence tables"
        ],
        rightAnswer: "B. Partial dependence plots (PDPs)"
    },
    {
        question: "A law firm wants to build an AI application by using large language models (LLMs). The application will read legal documents and extract key points from the documents. Which solution meets these requirements?",
        options: [
            "A. Build an automatic named entity recognition system.",
            "B. Create a recommendation engine.",
            "C. Develop a summarization chatbot.",
            "D. Develop a multi-language translation system."
        ],
        rightAnswer: "C. Develop a summarization chatbot."
    },
    {
        question: "A company wants to classify human genes into 20 categories based on gene characteristics. The company needs an ML algorithm to document how the inner mechanism of the model affects the output. Which ML algorithm meets these requirements?",
        options: [
            "A. Decision trees",
            "B. Linear regression",
            "C. Logistic regression",
            "D. Neural networks"
        ],
        rightAnswer: "A. Decision trees"
    },
    {
        question: "A company has built an image classification model to predict plant diseases from photos of plant leaves. The company wants to evaluate how many images the model classified correctly. Which evaluation metric should the company use to measure the model's performance?",
        options: [
            "A. R-squared score",
            "B. Accuracy",
            "C. Root mean squared error (RMSE)",
            "D. Learning rate"
        ],
        rightAnswer: "B. Accuracy"
    },
    {
        question: "A company is using a pre-trained large language model (LLM) to build a chatbot for product recommendations. The company needs the LLM outputs to be short and written in a specific language. Which solution will align the LLM response quality with the company's expectations?",
        options: [
            "A. Adjust the prompt.",
            "B. Choose an LLM of a different size.",
            "C. Increase the temperature.",
            "D. Increase the Top K value."
        ],
        rightAnswer: "A. Adjust the prompt."
    },
    {
        question: "A company uses Amazon SageMaker for its ML pipeline in a production environment. The company has large input data sizes up to 1 GB and processing times up to 1 hour. The company needs near real-time latency. Which SageMaker inference option meets these requirements?",
        options: [
            "A. Real-time inference",
            "B. Serverless inference",
            "C. Asynchronous inference",
            "D. Batch transform"
        ],
        rightAnswer: "C. Asynchronous inference"
    },
    {
        question: "A company is using domain-specific models. The company wants to avoid creating new models from the beginning. The company instead wants to adapt pre-trained models to create models for new, related tasks. Which ML strategy meets these requirements?",
        options: [
            "A. Increase the number of epochs.",
            "B. Use transfer learning.",
            "C. Decrease the number of epochs.",
            "D. Use unsupervised learning."
        ],
        rightAnswer: "B. Use transfer learning."
    },
    {
        question: "A company is building a solution to generate images for protective eyewear. The solution must have high accuracy and must minimize the risk of incorrect annotations. Which solution will meet these requirements?",
        options: [
            "A. Human-in-the-loop validation by using Amazon SageMaker Ground Truth Plus",
            "B. Data augmentation by using an Amazon Bedrock knowledge base",
            "C. Image recognition by using Amazon Rekognition",
            "D. Data summarization by using Amazon QuickSight Q"
        ],
        rightAnswer: "A. Human-in-the-loop validation by using Amazon SageMaker Ground Truth Plus"
    },
    {
        question: "A company wants to create a chatbot by using a foundation model (FM) on Amazon Bedrock. The FM needs to access encrypted data that is stored in an Amazon S3 bucket. The data is encrypted with Amazon S3 managed keys (SSE-S3). The FM encounters a failure when attempting to access the S3 bucket data. Which solution will meet these requirements?",
        options: [
            "A. Ensure that the role that Amazon Bedrock assumes has permission to decrypt data with the correct encryption key.",
            "B. Set the access permissions for the S3 buckets to allow public access to enable access over the internet.",
            "C. Use prompt engineering techniques to tell the model to look for information in Amazon S3.",
            "D. Ensure that the S3 data does not contain sensitive information."
        ],
        rightAnswer: "A. Ensure that the role that Amazon Bedrock assumes has permission to decrypt data with the correct encryption key."
    },
    {
        question: "A company wants to use language models to create an application for inference on edge devices. The inference must have the lowest latency possible. Which solution will meet these requirements?",
        options: [
            "A. Deploy optimized small language models (SLMs) on edge devices.",
            "B. Deploy optimized large language models (LLMs) on edge devices.",
            "C. Incorporate a centralized small language model (SLM) API for asynchronous communication with edge devices.",
            "D. Incorporate a centralized large language model (LLM) API for asynchronous communication with edge devices."
        ],
        rightAnswer: "A. Deploy optimized small language models (SLMs) on edge devices."
    },

    {
        question: "A financial institution is using Amazon Bedrock to develop an AI application. The application is hosted in a VPC. To meet regulatory compliance standards, the VPC is not allowed access to any internet traffic. Which AWS service or feature will meet these requirements?",
        options: [
            "A. AWS PrivateLink",
            "B. Amazon Macie",
            "C. Amazon CloudFront",
            "D. Internet gateway"
        ],
        rightAnswer: "A. AWS PrivateLink"
    },
    {
        question: "A company wants to develop an educational game where users answer questions such as the following: 'A jar contains six red, four green, and three yellow marbles. What is the probability of choosing a green marble from the jar?' Which solution meets these requirements with the LEAST operational overhead?",
        options: [
            "A. Use supervised learning to create a regression model that will predict probability.",
            "B. Use reinforcement learning to train a model to return the probability.",
            "C. Use code that will calculate probability by using simple rules and computations.",
            "D. Use unsupervised learning to create a model that will estimate probability density."
        ],
        rightAnswer: "C. Use code that will calculate probability by using simple rules and computations."
    },
    {
        question: "Which metric measures the runtime efficiency of operating AI models?",
        options: [
            "A. Customer satisfaction score (CSAT)",
            "B. Training time for each epoch",
            "C. Average response time",
            "D. Number of training instances"
        ],
        rightAnswer: "C. Average response time"
    },
    {
        question: "A company is building a contact center application and wants to gain insights from customer conversations. The company wants to analyze and extract key information from the audio of the customer calls. Which solution meets these requirements?",
        options: [
            "A. Build a conversational chatbot by using Amazon Lex.",
            "B. Transcribe call recordings by using Amazon Transcribe.",
            "C. Extract information from call recordings by using Amazon SageMaker Model Monitor.",
            "D. Create classification labels by using Amazon Comprehend."
        ],
        rightAnswer: "B. Transcribe call recordings by using Amazon Transcribe."
    },
    {
        question: "An AI practitioner wants to use a foundation model (FM) to design a search application. The search application must handle queries that have text and images. Which type of FM should the AI practitioner use to power the search application?",
        options: [
            "A. Multi-modal embedding model",
            "B. Text embedding model",
            "C. Multi-modal generation model",
            "D. Image generation model"
        ],
        rightAnswer: "A. Multi-modal embedding model"
    },
    {
        question: "Which feature of Amazon OpenSearch Service gives companies the ability to build vector database applications?",
        options: [
            "A. Integration with Amazon S3 for object storage",
            "B. Support for geospatial indexing and queries",
            "C. Scalable index management and nearest neighbor search capability",
            "D. Ability to perform real-time analysis on streaming data"
        ],
        rightAnswer: "C. Scalable index management and nearest neighbor search capability"
    },
    {
        question: "Which option is a use case for generative AI models?",
        options: [
            "A. Improving network security by using intrusion detection systems",
            "B. Creating photorealistic images from text descriptions for digital marketing",
            "C. Enhancing database performance by using optimized indexing",
            "D. Analyzing financial data to forecast stock market trends"
        ],
        rightAnswer: "B. Creating photorealistic images from text descriptions for digital marketing"
    },
    {
        question: "A company wants to build a generative AI application by using Amazon Bedrock and needs to choose a foundation model (FM). The company wants to know how much information can fit into one prompt. Which consideration will inform the company's decision?",
        options: [
            "A. Temperature",
            "B. Context window",
            "C. Batch size",
            "D. Model size"
        ],
        rightAnswer: "B. Context window"
    },
    {
        question: "A company wants to make a chatbot to help customers. The chatbot will help solve technical problems without human intervention. The company chose a foundation model (FM) for the chatbot. The chatbot needs to produce responses that adhere to company tone. Which solution meets these requirements?",
        options: [
            "A. Set a low limit on the number of tokens the FM can produce.",
            "B. Use batch inferencing to process detailed responses.",
            "C. Experiment and refine the prompt until the FM produces the desired responses.",
            "D. Define a higher number for the temperature parameter."
        ],
        rightAnswer: "C. Experiment and refine the prompt until the FM produces the desired responses."
    },
    {
        question: "A company wants to use a large language model (LLM) on Amazon Bedrock for sentiment analysis. The company wants to classify the sentiment of text passages as positive or negative. Which prompt engineering strategy meets these requirements?",
        options: [
            "A. Provide examples of text passages with corresponding positive or negative labels in the prompt followed by the new text passage to be classified.",
            "B. Provide a detailed explanation of sentiment analysis and how LLMs work in the prompt.",
            "C. Provide the new text passage to be classified without any additional context or examples.",
            "D. Provide the new text passage with a few examples of unrelated tasks, such as text summarization or question answering."
        ],
        rightAnswer: "A. Provide examples of text passages with corresponding positive or negative labels in the prompt followed by the new text passage to be classified."
    },
    {
        question: "A security company is using Amazon Bedrock to run foundation models (FMs). The company wants to ensure that only authorized users invoke the models. The company needs to identify any unauthorized access attempts to set appropriate AWS Identity and Access Management (IAM) policies and roles for future iterations of the FMs. Which AWS service should the company use to identify unauthorized users that are trying to access Amazon Bedrock?",
        options: [
            "A. AWS Audit Manager",
            "B. AWS CloudTrail",
            "C. Amazon Fraud Detector",
            "D. AWS Trusted Advisor"
        ],
        rightAnswer: "B. AWS CloudTrail"
    },
    {
        question: "A company wants to use a large language model (LLM) to develop a conversational agent. The company needs to prevent the LLM from being manipulated with common prompt engineering techniques to perform undesirable actions or expose sensitive information. Which action will reduce these risks?",
        options: [
            "A. Create a prompt template that teaches the LLM to detect attack patterns.",
            "B. Increase the temperature parameter on invocation requests to the LLM.",
            "C. Avoid using LLMs that are not listed in Amazon SageMaker.",
            "D. Decrease the number of input tokens on invocations of the LLM."
        ],
        rightAnswer: "A. Create a prompt template that teaches the LLM to detect attack patterns."
    },
    {
        question: "A company is using the Generative AI Security Scoping Matrix to assess security responsibilities for its solutions. The company has identified four different solution scopes based on the matrix. Which solution scope gives the company the MOST ownership of security responsibilities?",
        options: [
            "A. Using a third-party enterprise application that has embedded generative AI features.",
            "B. Building an application by using an existing third-party generative AI foundation model (FM).",
            "C. Refining an existing third-party generative AI foundation model (FM) by fine-tuning the model by using data specific to the business.",
            "D. Building and training a generative AI model from scratch by using specific data that a customer owns."
        ],
        rightAnswer: "D. Building and training a generative AI model from scratch by using specific data that a customer owns."
    },
    {
        question: "A company wants to create an application by using Amazon Bedrock. The company has a limited budget and prefers flexibility without long-term commitment. Which Amazon Bedrock pricing model meets these requirements?",
        options: [
            "A. On-Demand",
            "B. Model customization",
            "C. Provisioned Throughput",
            "D. Spot Instance"
        ],
        rightAnswer: "A. On-Demand"
    },
    {
        question: "Which AWS service or feature can help an AI development team quickly deploy and consume a foundation model (FM) within the team's VPC?",
        options: [
            "A. Amazon Personalize",
            "B. Amazon SageMaker JumpStart",
            "C. PartyRock, an Amazon Bedrock Playground",
            "D. Amazon SageMaker endpoints"
        ],
        rightAnswer: "B. Amazon SageMaker JumpStart"
    },
    {
        question: "How can companies use large language models (LLMs) securely on Amazon Bedrock?",
        options: [
            "A. Design clear and specific prompts. Configure AWS Identity and Access Management (IAM) roles and policies by using least privilege access.",
            "B. Enable AWS Audit Manager for automatic model evaluation jobs.",
            "C. Enable Amazon Bedrock automatic model evaluation jobs.",
            "D. Use Amazon CloudWatch Logs to make models explainable and to monitor for bias."
        ],
        rightAnswer: "A. Design clear and specific prompts. Configure AWS Identity and Access Management (IAM) roles and policies by using least privilege access."
    },
    {
        question: "A company has terabytes of data in a database that the company can use for business analysis. The company wants to build an AI-based application that can build a SQL query from input text that employees provide. The employees have minimal experience with technology. Which solution meets these requirements?",
        options: [
            "A. Generative pre-trained transformers (GPT)",
            "B. Residual neural network",
            "C. Support vector machine",
            "D. WaveNet"
        ],
        rightAnswer: "A. Generative pre-trained transformers (GPT)"
    },
    {
        question: "A company built a deep learning model for object detection and deployed the model to production. Which AI process occurs when the model analyzes a new image to identify objects?",
        options: [
            "A. Training",
            "B. Inference",
            "C. Model deployment",
            "D. Bias correction"
        ],
        rightAnswer: "B. Inference"
    },
    {
        question: "An AI practitioner is building a model to generate images of humans in various professions. The AI practitioner discovered that the input data is biased and that specific attributes affect the image generation and create bias in the model. Which technique will solve the problem?",
        options: [
            "A. Data augmentation for imbalanced classes",
            "B. Model monitoring for class distribution",
            "C. Retrieval Augmented Generation (RAG)",
            "D. Watermark detection for images"
        ],
        rightAnswer: "A. Data augmentation for imbalanced classes"
    },
    {
        question: "A company is implementing the Amazon Titan foundation model (FM) by using Amazon Bedrock. The company needs to supplement the model by using relevant data from the company's private data sources. Which solution will meet this requirement?",
        options: [
            "A. Use a different FM.",
            "B. Choose a lower temperature value.",
            "C. Create an Amazon Bedrock knowledge base.",
            "D. Enable model invocation logging."
        ],
        rightAnswer: "C. Create an Amazon Bedrock knowledge base."
    },
    {
        question: "A medical company is customizing a foundation model (FM) for diagnostic purposes. The company needs the model to be transparent and explainable to meet regulatory requirements. Which solution will meet these requirements?",
        options: [
            "A. Configure the security and compliance by using Amazon Inspector.",
            "B. Generate simple metrics, reports, and examples by using Amazon SageMaker Clarify.",
            "C. Encrypt and secure training data by using Amazon Macie.",
            "D. Gather more data. Use Amazon Rekognition to add custom labels to the data."
        ],
        rightAnswer: "B. Generate simple metrics, reports, and examples by using Amazon SageMaker Clarify."
    },
    {
        question: "A company wants to deploy a conversational chatbot to answer customer questions. The chatbot is based on a fine-tuned Amazon SageMaker JumpStart model. The application must comply with multiple regulatory frameworks. Which capabilities can the company show compliance for? (Choose two.)",
        options: [
            "A. Auto scaling inference endpoints",
            "B. Threat detection",
            "C. Data protection",
            "D. Cost optimization",
            "E. Loosely coupled microservices"
        ],
        rightAnswer: ["B. Threat detection", "C. Data protection"]
    },
    {
        question: "A company is training a foundation model (FM). The company wants to increase the accuracy of the model up to a specific acceptance level. Which solution will meet these requirements?",
        options: [
            "A. Decrease the batch size.",
            "B. Increase the epochs.",
            "C. Decrease the epochs.",
            "D. Increase the temperature parameter."
        ],
        rightAnswer: "B. Increase the epochs."
    },
    {
        question: "An ecommerce company wants to build a solution to determine customer sentiments based on written customer reviews of products. Which AWS services meet these requirements? (Choose two.)",
        options: [
            "A. Amazon Lex",
            "B. Amazon Comprehend",
            "C. Amazon Polly",
            "D. Amazon Bedrock",
            "E. Amazon Rekognition"
        ],
        rightAnswer: ["B. Amazon Comprehend", "D. Amazon Bedrock"]
    },
    {
        question: "A company wants to use large language models (LLMs) with Amazon Bedrock to develop a chat interface for the company's product manuals. The manuals are stored as PDF files. Which solution meets these requirements MOST cost-effectively?",
        options: [
            "A. Use prompt engineering to add one PDF file as context to the user prompt when the prompt is submitted to Amazon Bedrock.",
            "B. Use prompt engineering to add all the PDF files as context to the user prompt when the prompt is submitted to Amazon Bedrock.",
            "C. Use all the PDF documents to fine-tune a model with Amazon Bedrock. Use the fine-tuned model to process user prompts.",
            "D. Upload PDF documents to an Amazon Bedrock knowledge base. Use the knowledge base to provide context when users submit prompts to Amazon Bedrock."
        ],
        rightAnswer: "D. Upload PDF documents to an Amazon Bedrock knowledge base. Use the knowledge base to provide context when users submit prompts to Amazon Bedrock."
    },
    {
        question: "A social media company wants to use a large language model (LLM) for content moderation. The company wants to evaluate the LLM outputs for bias and potential discrimination against specific groups or individuals. Which data source should the company use to evaluate the LLM outputs with the LEAST administrative effort?",
        options: [
            "A. User-generated content",
            "B. Moderation logs",
            "C. Content moderation guidelines",
            "D. Benchmark datasets"
        ],
        rightAnswer: "D. Benchmark datasets"
    },
    {
        question: "A company is using an Amazon Bedrock base model to summarize documents for an internal use case. The company trained a custom model to improve the summarization quality. Which action must the company take to use the custom model through Amazon Bedrock?",
        options: [
            "A. Purchase Provisioned Throughput for the custom model.",
            "B. Deploy the custom model in an Amazon SageMaker endpoint for real-time inference.",
            "C. Register the model with the Amazon SageMaker Model Registry.",
            "D. Grant access to the custom model in Amazon Bedrock."
        ],
        rightAnswer: "A. Purchase Provisioned Throughput for the custom model."
    },
    {
        question: "A digital devices company wants to predict customer demand for memory hardware. The company does not have coding experience or knowledge of ML algorithms and needs to develop a data-driven predictive model. The company needs to perform analysis on internal data and external data. Which solution will meet these requirements?",
        options: [
            "A. Store the data in Amazon S3. Create ML models and demand forecast predictions by using Amazon SageMaker built-in algorithms that use the data from Amazon S3.",
            "B. Import the data into Amazon SageMaker Data Wrangler. Create ML models and demand forecast predictions by using SageMaker built-in algorithms.",
            "C. Import the data into Amazon SageMaker Data Wrangler. Build ML models and demand forecast predictions by using an Amazon Personalize Trending-Now recipe.",
            "D. Import the data into Amazon SageMaker Canvas. Build ML models and demand forecast predictions by selecting the values in the data from SageMaker Canvas."
        ],
        rightAnswer: "D. Import the data into Amazon SageMaker Canvas. Build ML models and demand forecast predictions by selecting the values in the data from SageMaker Canvas."
    },
    {
        question: "A company has installed a security camera. The company uses an ML model to evaluate the security camera footage for potential thefts. The company has discovered that the model disproportionately flags people who are members of a specific ethnic group. Which type of bias is affecting the model output?",
        options: [
            "A. Measurement bias",
            "B. Sampling bias",
            "C. Observer bias",
            "D. Confirmation bias"
        ],
        rightAnswer: "B. Sampling bias"
    },
    {
        question: "A company is building a customer service chatbot. The company wants the chatbot to improve its responses by learning from past interactions and online resources. Which AI learning strategy provides this self-improvement capability?",
        options: [
            "A. Supervised learning with a manually curated dataset of good responses and bad responses",
            "B. Reinforcement learning with rewards for positive customer feedback",
            "C. Unsupervised learning to find clusters of similar customer inquiries",
            "D. Supervised learning with a continuously updated FAQ database"
        ],
        rightAnswer: "B. Reinforcement learning with rewards for positive customer feedback"
    },
    {
        question: "An AI practitioner has built a deep learning model to classify the types of materials in images. The AI practitioner now wants to measure the model performance. Which metric will help the AI practitioner evaluate the performance of the model?",
        options: [
            "A. Confusion matrix",
            "B. Correlation matrix",
            "C. R2 score",
            "D. Mean squared error (MSE)"
        ],
        rightAnswer: "A. Confusion matrix"
    },
    {
        question: "A company has built a chatbot that can respond to natural language questions with images. The company wants to ensure that the chatbot does not return inappropriate or unwanted images. Which solution will meet these requirements?",
        options: [
            "A. Implement moderation APIs.",
            "B. Retrain the model with a general public dataset.",
            "C. Perform model validation.",
            "D. Automate user feedback integration."
        ],
        rightAnswer: "A. Implement moderation APIs."
    },
    {
        question: "A company is building an ML model to analyze archived data. The company must perform inference on large datasets that are multiple GBs in size. The company does not need to access the model predictions immediately. Which Amazon SageMaker inference option will meet these requirements?",
        options: [
            "A. Batch transform",
            "B. Real-time inference",
            "C. Serverless inference",
            "D. Asynchronous inference"
        ],
        rightAnswer: "A. Batch transform"
    },
    {
        question: "A retail company intends to use machine learning to categorize new products. A labeled dataset of current products was provided to the Data Science team. The dataset includes 1,200 products. The labeled dataset has 15 features for each product such as title dimensions, weight, and price. Each product is labeled as belonging to one of six categories such as books, games, electronics, and movies. Which model should be used for categorizing new products using the provided dataset for training?",
        options: [
            "A. AnXGBoost model where the objective parameter is set to multi:softmax",
            "B. A deep convolutional neural network (CNN) with a softmax activation function for the last layer",
            "C. A regression forest where the number of trees is set equal to the number of product categories",
            "D. A DeepAR forecasting model based on a recurrent neural network (RNN)"
        ],
        rightAnswer: "A. AnXGBoost model where the objective parameter is set to multi:softmax"
    },
    {
        question: "A research company implemented a chatbot by using a foundation model (FM) from Amazon Bedrock. The chatbot searches for answers to questions from a large database of research papers. After multiple prompt engineering attempts, the company notices that the FM is performing poorly because of the complex scientific terms in the research papers. How can the company improve the performance of the chatbot?",
        options: [
            "A. Use few-shot prompting to define how the FM can answer the questions.",
            "B. Use domain adaptation fine-tuning to adapt the FM to complex scientific terms.",
            "C. Change the FM inference parameters.",
            "D. Clean the research paper data to remove complex scientific terms"
        ],
        rightAnswer: "B. Use domain adaptation fine-tuning to adapt the FM to complex scientific terms."
    },
    {
        question: "A company wants to develop a large language model (LLM) application by using Amazon Bedrock and customer data that is uploaded to Amazon S3. The company's security policy states that each team can access data for only the team's own customers. Which solution will meet these requirements?",
        options: [
            "A. Create an Amazon Bedrock custom service role for each team that has access to only the team's customer data.",
            "B. Create a custom service role that has Amazon S3 access. Ask teams to specify the customer name on each Amazon Bedrock request.",
            "C. Redact personal data in Amazon S3. Update the S3 bucket policy to allow team access to customer data.",
            "D. Create one Amazon Bedrock role that has full Amazon S3 access. Create IAM roles for each team that have access to only each team's customer folders."
        ],
        rightAnswer: "A. Create an Amazon Bedrock custom service role for each team that has access to only the team's customer data."
    },
    {
        question: "A medical company deployed a disease detection model on Amazon Bedrock. To comply with privacy policies, the company wants to prevent the model from including personal patient information in its responses. The company also wants to receive notification when policy violations occur. Which solution meets these requirements?",
        options: [
            "A. Use Amazon Macie to scan the model's output for sensitive data and set up alerts for potential violations.",
            "B. Configure AWS CloudTrail to monitor the model's responses and create alerts for any detected personal information.",
            "C. Use Guardrails for Amazon Bedrock to filter content. Set up Amazon CloudWatch alarms for notification of policy violations.",
            "D. Implement Amazon SageMaker Model Monitor to detect data drift and receive alerts when model quality degrades."
        ],
        rightAnswer: "C. Use Guardrails for Amazon Bedrock to filter content. Set up Amazon CloudWatch alarms for notification of policy violations."
    },
    {
        question: "A company manually reviews all submitted resumes in PDF format. As the company grows, the company expects the volume of resumes to exceed the company's review capacity. The company needs an automated system to convert the PDF resumes into plain text format for additional processing. Which AWS service meets this requirement?",
        options: [
            "A. Amazon Textract",
            "B. Amazon Personalize",
            "C. Amazon Lex",
            "D. Amazon Transcribe"
        ],
        rightAnswer: "A. Amazon Textract"
    },
    {
        question: "An education provider is building a question and answer application that uses a generative AI model to explain complex concepts. The education provider wants to automatically change the style of the model response depending on who is asking the question. The education provider will give the model the age range of the user who has asked the question. Which solution meets these requirements with the LEAST implementation effort?",
        options: [
            "A. Fine-tune the model by using additional training data that is representative of the various age ranges that the application will support.",
            "B. Add a role description to the prompt context that instructs the model of the age range that the response should target.",
            "C. Use chain-of-thought reasoning to deduce the correct style and complexity for a response suitable for that user.",
            "D. Summarize the response text depending on the age of the user so that younger users receive shorter responses."
        ],
        rightAnswer: "B. Add a role description to the prompt context that instructs the model of the age range that the response should target."
    },
    {
        question: "Which strategy evaluates the accuracy of a foundation model (FM) that is used in image classification tasks?",
        options: [
            "A. Calculate the total cost of resources used by the model.",
            "B. Measure the model's accuracy against a predefined benchmark dataset.",
            "C. Count the number of layers in the neural network.",
            "D. Assess the color accuracy of images processed by the model."
        ],
        rightAnswer: "B. Measure the model's accuracy against a predefined benchmark dataset."
    },
    {
        question: "An accounting firm wants to implement a large language model (LLM) to automate document processing. The firm must proceed responsibly to avoid potential harms. What should the firm do when developing and deploying the LLM? (Choose two.)",
        options: [
            "A. Include fairness metrics for model evaluation.",
            "B. Adjust the temperature parameter of the model.",
            "C. Modify the training data to mitigate bias.",
            "D. Avoid overfitting on the training data.",
            "E. Apply prompt engineering techniques."
        ],
        rightAnswer: ["A. Include fairness metrics for model evaluation.", "C. Modify the training data to mitigate bias."]
    },
    {
        question: "A company is building an ML model. The company collected new data and analyzed the data by creating a correlation matrix, calculating statistics, and visualizing the data. Which stage of the ML pipeline is the company currently in?",
        options: [
            "A. Data pre-processing",
            "B. Feature engineering",
            "C. Exploratory data analysis",
            "D. Hyperparameter tuning"
        ],
        rightAnswer: "C. Exploratory data analysis"
    },
    {
        question: "A company has documents that are missing some words because of a database error. The company wants to build an ML model that can suggest potential words to fill in the missing text. Which type of model meets this requirement?",
        options: [
            "A. Topic modeling",
            "B. Clustering models",
            "C. Prescriptive ML models",
            "D. BERT-based models"
        ],
        rightAnswer: "D. BERT-based models"
    },
    {
        question: "A company wants to display the total sales for its top-selling products across various retail locations in the past 12 months. Which AWS solution should the company use to automate the generation of graphs?",
        options: [
            "A. Amazon Q in Amazon EC2",
            "B. Amazon Q Developer",
            "C. Amazon Q in Amazon QuickSight",
            "D. Amazon Q in AWS Chatbot"
        ],
        rightAnswer: "C. Amazon Q in Amazon QuickSight"
    },
    {
        question: "A company is building a chatbot to improve user experience. The company is using a large language model (LLM) from Amazon Bedrock for intent detection. The company wants to use few-shot learning to improve intent detection accuracy. Which additional data does the company need to meet these requirements?",
        options: [
            "A. Pairs of chatbot responses and correct user intents",
            "B. Pairs of user messages and correct chatbot responses",
            "C. Pairs of user messages and correct user intents",
            "D. Pairs of user intents and correct chatbot responses"
        ],
        rightAnswer: "C. Pairs of user messages and correct user intents"
    },
    {
        question: "A company is using few-shot prompting on a base model that is hosted on Amazon Bedrock. The model currently uses 10 examples in the prompt. The model is invoked once daily and is performing well. The company wants to lower the monthly cost. Which solution will meet these requirements?",
        options: [
            "A. Customize the model by using fine-tuning.",
            "B. Decrease the number of tokens in the prompt.",
            "C. Increase the number of tokens in the prompt.",
            "D. Use Provisioned Throughput."
        ],
        rightAnswer: "B. Decrease the number of tokens in the prompt."
    },
    {
        question: "An AI practitioner is using a large language model (LLM) to create content for marketing campaigns. The generated content sounds plausible and factual but is incorrect. Which problem is the LLM having?",
        options: [
            "A. Data leakage",
            "B. Hallucination",
            "C. Overfitting",
            "D. Underfitting"
        ],
        rightAnswer: "B. Hallucination"
    },
    {
        question: "An AI practitioner trained a custom model on Amazon Bedrock by using a training dataset that contains confidential data. The AI practitioner wants to ensure that the custom model does not generate inference responses based on confidential data. How should the AI practitioner prevent responses based on confidential data?",
        options: [
            "A. Delete the custom model. Remove the confidential data from the training dataset. Retrain the custom model.",
            "B. Mask the confidential data in the inference responses by using dynamic data masking.",
            "C. Encrypt the confidential data in the inference responses by using Amazon SageMaker.",
            "D. Encrypt the confidential data in the custom model by using AWS Key Management Service (AWS KMS)."
        ],
        rightAnswer: "A. Delete the custom model. Remove the confidential data from the training dataset. Retrain the custom model."
    },
    {
        question: "A company has built a solution by using generative AI. The solution uses large language models (LLMs) to translate training manuals from English into other languages. The company wants to evaluate the accuracy of the solution by examining the text generated for the manuals. Which model evaluation strategy meets these requirements?",
        options: [
            "A. Bilingual Evaluation Understudy (BLEU)",
            "B. Root mean squared error (RMSE)",
            "C. Recall-Oriented Understudy for Gisting Evaluation (ROUGE)",
            "D. F1 score"
        ],
        rightAnswer: "A. Bilingual Evaluation Understudy (BLEU)"
    },
    {
        question: "A large retailer receives thousands of customer support inquiries about products every day. The customer support inquiries need to be processed and responded too quickly. The company wants to implement Agents for Amazon Bedrock. What are the key benefits of using Amazon Bedrock agents that could help this retailer?",
        options: [
            "A. Generation of custom foundation models (FMs) to predict customer needs",
            "B. Automation of repetitive tasks and orchestration of complex workflows",
            "C. Automatically calling multiple foundation models (FMs) and consolidating the results",
            "D. Selecting the foundation model (FM) based on predefined criteria and metrics"
        ],
        rightAnswer: "B. Automation of repetitive tasks and orchestration of complex workflows"
    },
    {
        question: "Which option is a benefit of ongoing pre-training when fine-tuning a foundation model (FM)?",
        options: [
            "A. Helps decrease the model's complexity",
            "B. Improves model performance over time",
            "C. Decreases the training time requirement",
            "D. Optimizes model inference time"
        ],
        rightAnswer: "B. Improves model performance over time"
    },
    {
        question: "What are tokens in the context of generative AI models?",
        options: [
            "A. Tokens are the basic units of input and output that a generative AI model operates on, representing words, subwords, or other linguistic units.",
            "B. Tokens are the mathematical representations of words or concepts used in generative AI models.",
            "C. Tokens are the pre-trained weights of a generative AI model that are fine-tuned for specific tasks.",
            "D. Tokens are the specific prompts or instructions given to a generative AI model to generate output."
        ],
        rightAnswer: "A. Tokens are the basic units of input and output that a generative AI model operates on, representing words, subwords, or other linguistic units."
    },
    {
        question: "A company wants to assess the costs that are associated with using a large language model (LLM) to generate inferences. The company wants to use Amazon Bedrock to build generative AI applications. Which factor will drive the inference costs?",
        options: [
            "A. Number of tokens consumed",
            "B. Temperature value",
            "C. Amount of data used to train the LLM",
            "D. Total training time"
        ],
        rightAnswer: "A. Number of tokens consumed"
    },
    {
        question: "A company is using Amazon SageMaker Studio notebooks to build and train ML models. The company stores the data in an Amazon S3 bucket. The company needs to manage the flow of data from Amazon S3 to SageMaker Studio notebooks. Which solution will meet this requirement?",
        options: [
            "A. Use Amazon Inspector to monitor SageMaker Studio.",
            "B. Use Amazon Macie to monitor SageMaker Studio.",
            "C. Configure SageMaker to use a VPC with an S3 endpoint.",
            "D. Configure SageMaker to use S3 Glacier Deep Archive."
        ],
        rightAnswer: "C. Configure SageMaker to use a VPC with an S3 endpoint."
    },
    {
        question: "A company has a foundation model (FM) that was customized by using Amazon Bedrock to answer customer queries about products. The company wants to validate the model's responses to new types of queries. The company needs to upload a new dataset that Amazon Bedrock can use for validation. Which AWS service meets these requirements?",
        options: [
            "A. Amazon S3",
            "B. Amazon Elastic Block Store (Amazon EBS)",
            "C. Amazon Elastic File System (Amazon EFS)",
            "D. AWS Snowcone"
        ],
        rightAnswer: "A. Amazon S3"
    },
    {
        question: "Which prompting attack directly exposes the configured behavior of a large language model (LLM)?",
        options: [
            "A. Prompted persona switches",
            "B. Exploiting friendliness and trust",
            "C. Ignoring the prompt template",
            "D. Extracting the prompt template"
        ],
        rightAnswer: "D. Extracting the prompt template"
    },
    {
        question: "A company wants to use Amazon Bedrock. The company needs to review which security aspects the company is responsible for when using Amazon Bedrock. Which security aspect will the company be responsible for?",
        options: [
            "A. Patching and updating the versions of Amazon Bedrock",
            "B. Protecting the infrastructure that hosts Amazon Bedrock",
            "C. Securing the company's data in transit and at rest",
            "D. Provisioning Amazon Bedrock within the company network"
        ],
        rightAnswer: "C. Securing the company's data in transit and at rest"
    },
    {
        question: "A social media company wants to use a large language model (LLM) to summarize messages. The company has chosen a few LLMs that are available on Amazon SageMaker JumpStart. The company wants to compare the generated output toxicity of these models. Which strategy gives the company the ability to evaluate the LLMs with the LEAST operational overhead?",
        options: [
            "A. Crowd-sourced evaluation",
            "B. Automatic model evaluation",
            "C. Model evaluation with human workers",
            "D. Reinforcement learning from human feedback (RLHF)"
        ],
        rightAnswer: "B. Automatic model evaluation"
    },
    {
        question: "A company is testing the security of a foundation model (FM). During testing, the company wants to get around the safety features and make harmful content. Which security technique is this an example of?",
        options: [
            "A. Fuzzing training data to find vulnerabilities",
            "B. Denial of service (DoS)",
            "C. Penetration testing with authorization",
            "D. Jailbreak"
        ],
        rightAnswer: "D. Jailbreak"
    },
    {
        question: "A company needs to use Amazon SageMaker for model training and inference. The company must comply with regulatory requirements to run SageMaker jobs in an isolated environment without internet access. Which solution will meet these requirements?",
        options: [
            "A. Run SageMaker training and inference by using SageMaker Experiments.",
            "B. Run SageMaker training and Inference by using network Isolation.",
            "C. Encrypt the data at rest by using encryption for SageMaker geospatial capabilities.",
            "D. Associate appropriate AWS Identity and Access Management (IAM) roles with the SageMaker jobs."
        ],
        rightAnswer: "B. Run SageMaker training and Inference by using network Isolation."
    },
    {
        question: "An ML research team develops custom ML models. The model artifacts are shared with other teams for integration into products and services. The ML team retains the model training code and data. The ML team wants to build a mechanism that the ML team can use to audit models. Which solution should the ML team use when publishing the custom ML models?",
        options: [
            "A. Create documents with the relevant information. Store the documents in Amazon S3.",
            "B. Use AWS AI Service Cards for transparency and understanding models.",
            "C. Create Amazon SageMaker Model Cards with intended uses and training and inference details.",
            "D. Create model training scripts. Commit the model training scripts to a Git repository."
        ],
        rightAnswer: "C. Create Amazon SageMaker Model Cards with intended uses and training and inference details."
    },
    {
        question: "A software company builds tools for customers. The company wants to use AI to increase software development productivity. Which solution will meet these requirements?",
        options: [
            "A. Use a binary classification model to generate code reviews.",
            "B. Install code recommendation software in the company's developer tools.",
            "C. Install a code forecasting tool to predict potential code issues.",
            "D. Use a natural language processing (NLP) tool to generate code."
        ],
        rightAnswer: "B. Install code recommendation software in the company's developer tools."
    },
    {
        question: "A retail store wants to predict the demand for a specific product for the next few weeks by using the Amazon SageMaker DeepAR forecasting algorithm. Which type of data will meet this requirement?",
        options: [
            "A. Text data",
            "B. Image data",
            "C. Time series data",
            "D. Binary data"
        ],
        rightAnswer: "C. Time series data"
    },
    {
        question: "A large retail bank wants to develop an ML system to help the risk management team decide on loan allocations for different demographics. What must the bank do to develop an unbiased ML model?",
        options: [
            "A. Reduce the size of the training dataset.",
            "B. Ensure that the ML model predictions are consistent with historical results.",
            "C. Create a different ML model for each demographic group.",
            "D. Measure class imbalance on the training dataset. Adapt the training process accordingly."
        ],
        rightAnswer: "D. Measure class imbalance on the training dataset. Adapt the training process accordingly."
    },
    {
        question: "Which prompting technique can protect against prompt injection attacks?",
        options: [
            "A. Adversarial prompting",
            "B. Zero-shot prompting",
            "C. Least-to-most prompting",
            "D. Chain-of-thought prompting"
        ],
        rightAnswer: "A. Adversarial prompting"
    },
    {
        question: "A company has fine-tuned a large language model (LLM) to answer questions for a help desk. The company wants to determine if the fine-tuning has enhanced the model's accuracy. Which metric should the company use for the evaluation?",
        options: [
            "A. Precision",
            "B. Time to first token",
            "C. F1 score",
            "D. Word error rate"
        ],
        rightAnswer: "C. F1 score"
    },
    {
        question: "A company is using Retrieval Augmented Generation (RAG) with Amazon Bedrock and Stable Diffusion to generate product images based on text descriptions. The results are often random and lack specific details. The company wants to increase the specificity of the generated images. Which solution meets these requirements?",
        options: [
            "A. Increase the number of generation steps.",
            "B. Use the MASK_IMAGE_BLACK mask source option.",
            "C. Increase the classifier-free guidance (CFG) scale.",
            "D. Increase the prompt strength."
        ],
        rightAnswer: "C. Increase the classifier-free guidance (CFG) scale."
    },
    {
        question: "A company wants to implement a large language model (LLM) based chatbot to provide customer service agents with real-time contextual responses to customers' inquiries. The company will use the company's policies as the knowledge base. Which solution will meet these requirements MOST cost-effectively?",
        options: [
            "A. Retrain the LLM on the company policy data.",
            "B. Fine-tune the LLM on the company policy data.",
            "C. Implement Retrieval Augmented Generation (RAG) for in-context responses.",
            "D. Use pre-training and data augmentation on the company policy data."
        ],
        rightAnswer: "C. Implement Retrieval Augmented Generation (RAG) for in-context responses."
    },
    {
        question: "A company wants to create a new solution by using AWS Glue. The company has minimal programming experience with AWS Glue. Which AWS service can help the company use AWS Glue?",
        options: [
            "A. Amazon Q Developer",
            "B. AWS Config",
            "C. Amazon Personalize",
            "D. Amazon Comprehend"
        ],
        rightAnswer: "A. Amazon Q Developer"
    },
    {
        question: "A company is developing a mobile ML app that uses a phone's camera to diagnose and treat insect bites. The company wants to train an image classification model by using a diverse dataset of insect bite photos from different genders, ethnicities, and geographic locations around the world. Which principle of responsible AI does the company demonstrate in this scenario?",
        options: [
            "A. Fairness",
            "B. Explainability",
            "C. Governance",
            "D. Transparency"
        ],
        rightAnswer: "A. Fairness"
    },
    {
        question: "A company is developing an ML model to make loan approvals. The company must implement a solution to detect bias in the model. The company must also be able to explain the model's predictions. Which solution will meet these requirements?",
        options: [
            "A. Amazon SageMaker Clarify",
            "B. Amazon SageMaker Data Wrangler",
            "C. Amazon SageMaker Model Cards",
            "D. AWS AI Service Cards"
        ],
        rightAnswer: "A. Amazon SageMaker Clarify"
    },
    {
        question: "A company has developed a generative text summarization model by using Amazon Bedrock. The company will use Amazon Bedrock automatic model evaluation capabilities. Which metric should the company use to evaluate the accuracy of the model?",
        options: [
            "A. Area Under the ROC Curve (AUC) score",
            "B. F1 score",
            "C. BERTScore",
            "D. Real world knowledge (RWK) score"
        ],
        rightAnswer: "C. BERTScore"
    },
    {
        question: "An AI practitioner wants to predict the classification of flowers based on petal length, petal width, sepal length, and sepal width. Which algorithm meets these requirements?",
        options: [
            "A. K-nearest neighbors (k-NN)",
            "B. K-mean",
            "C. Autoregressive Integrated Moving Average (ARIMA)",
            "D. Linear regression"
        ],
        rightAnswer: "A. K-nearest neighbors (k-NN)"
    },
    {
        question: "A company is using custom models in Amazon Bedrock for a generative AI application. The company wants to use a company managed encryption key to encrypt the model artifacts that the model customization jobs create. Which AWS service meets these requirements?",
        options: [
            "A. AWS Key Management Service (AWS KMS)",
            "B. Amazon Inspector",
            "C. Amazon Macie",
            "D. AWS Secrets Manager"
        ],
        rightAnswer: "A. AWS Key Management Service (AWS KMS)"
    },
    {
        question: "A company wants to use large language models (LLMs) to produce code from natural language code comments. Which LLM feature meets these requirements?",
        options: [
            "A. Text summarization",
            "B. Text generation",
            "C. Text completion",
            "D. Text classification"
        ],
        rightAnswer: "B. Text generation"
    },
    {
        question: "A company is introducing a mobile app that helps users learn foreign languages. The app makes text more coherent by calling a large language model (LLM). The company collected a diverse dataset of text and supplemented the dataset with examples of more readable versions. The company wants the LLM output to resemble the provided examples. Which metric should the company use to assess whether the LLM meets these requirements?",
        options: [
            "A. Value of the loss function",
            "B. Semantic robustness",
            "C. Recall-Oriented Understudy for Gisting Evaluation (ROUGE) score",
            "D. Latency of the text generation"
        ],
        rightAnswer: "C. Recall-Oriented Understudy for Gisting Evaluation (ROUGE) score"
    },
    {
        question: "A company notices that its foundation model (FM) generates images that are unrelated to the prompts. The company wants to modify the prompt techniques to decrease unrelated images. Which solution meets these requirements?",
        options: [
            "A. Use zero-shot prompts.",
            "B. Use negative prompts.",
            "C. Use positive prompts.",
            "D. Use ambiguous prompts."
        ],
        rightAnswer: "B. Use negative prompts."
    },
    {
        question: "A company wants to use a large language model (LLM) to generate concise, feature-specific descriptions for the company's products. Which prompt engineering technique meets these requirements?",
        options: [
            "A. Create one prompt that covers all products. Edit the responses to make the responses more specific, concise, and tailored to each product.",
            "B. Create prompts for each product category that highlight the key features. Include the desired output format and length for each prompt response.",
            "C. Include a diverse range of product features in each prompt to generate creative and unique descriptions.",
            "D. Provide detailed, product-specific prompts to ensure precise and customized descriptions."
        ],
        rightAnswer: "B. Create prompts for each product category that highlight the key features. Include the desired output format and length for each prompt response."
    },
    {
        question: "A company is developing an ML model to predict customer churn. The model performs well on the training dataset but does not accurately predict churn for new data. Which solution will resolve this issue?",
        options: [
            "A. Decrease the regularization parameter to increase model complexity.",
            "B. Increase the regularization parameter to decrease model complexity.",
            "C. Add more features to the input data.",
            "D. Train the model for more epochs."
        ],
        rightAnswer: "B. Increase the regularization parameter to decrease model complexity."
    },
    {
        question: "A company is implementing intelligent agents to provide conversational search experiences for its customers. The company needs a database service that will support storage and queries of embeddings from a generative AI model as vectors in the database. Which AWS service will meet these requirements?",
        options: [
            "A. Amazon Athena",
            "B. Amazon Aurora PostgreSQL",
            "C. Amazon Redshift",
            "D. Amazon EMR"
        ],
        rightAnswer: "B. Amazon Aurora PostgreSQL"
    },
    {
        question: "A financial institution is building an AI solution to make loan approval decisions by using a foundation model (FM). For security and audit purposes, the company needs the AI solution's decisions to be explainable. Which factor relates to the explainability of the AI solution's decisions?",
        options: [
            "A. Model complexity",
            "B. Training time",
            "C. Number of hyperparameters",
            "D. Deployment time"
        ],
        rightAnswer: "A. Model complexity"
    },
    {
        question: "A pharmaceutical company wants to analyze user reviews of new medications and provide a concise overview for each medication. Which solution meets these requirements?",
        options: [
            "A. Create a time-series forecasting model to analyze the medication reviews by using Amazon Personalize.",
            "B. Create medication review summaries by using Amazon Bedrock large language models (LLMs).",
            "C. Create a classification model that categorizes medications into different groups by using Amazon SageMaker.",
            "D. Create medication review summaries by using Amazon Rekognition."
        ],
        rightAnswer: "B. Create medication review summaries by using Amazon Bedrock large language models (LLMs)."
    },
    {
        question: "A company wants to build a lead prioritization application for its employees to contact potential customers. The application must give employees the ability to view and adjust the weights assigned to different variables in the model based on domain knowledge and expertise. Which ML model type meets these requirements?",
        options: [
            "A. Logistic regression model",
            "B. Deep learning model built on principal components",
            "C. K-nearest neighbors (k-NN) model",
            "D. Neural network"
        ],
        rightAnswer: "A. Logistic regression model"
    },
    {
        question: "A company wants to build an ML application. Select and order the correct steps from the following list to develop a well-architected ML workload. Each step should be selected one time. List:\n\nA. Deploy model\nB. Develop model\nC. Monitor model\nD. Define business goal and frame ML problem",
        options: [
            "A. A, B, C, D",
            "B. D, B, A, C",
            "C. B, C, D, A",
            "D. C, A, B, D"
        ],
        rightAnswer: "B. D, B, A, C"
    }
    ,
    {
        question: "Which strategy will determine if a foundation model (FM) effectively meets business objectives?",
        options: [
            "A. Evaluate the model's performance on benchmark datasets.",
            "B. Analyze the model's architecture and hyperparameters.",
            "C. Assess the model's alignment with specific use cases.",
            "D. Measure the computational resources required for model deployment."
        ],
        rightAnswer: "C. Assess the model's alignment with specific use cases."
    },
    {
        question: "A company needs to train an ML model to classify images of different types of animals. The company has a large dataset of labeled images and will not label more data. Which type of learning should the company use to train the model?",
        options: [
            "A. Supervised learning",
            "B. Unsupervised learning",
            "C. Reinforcement learning",
            "D. Active learning"
        ],
        rightAnswer: "A. Supervised learning"
    },
    {
        question: "Which phase of the ML lifecycle determines compliance and regulatory requirements?",
        options: [
            "A. Feature engineering",
            "B. Model training",
            "C. Data collection",
            "D. Business goal identification"
        ],
        rightAnswer: "D. Business goal identification"
    },
    {
        question: "A food service company wants to develop an ML model to help decrease daily food waste and increase sales revenue. The company needs to continuously improve the model's accuracy. Which solution meets these requirements?",
        options: [
            "A. Use Amazon SageMaker and iterate with newer data.",
            "B. Use Amazon Personalize and iterate with historical data.",
            "C. Use Amazon CloudWatch to analyze customer orders.",
            "D. Use Amazon Rekognition to optimize the model."
        ],
        rightAnswer: "A. Use Amazon SageMaker and iterate with newer data."
    },
    {
        question: "A company has developed an ML model to predict real estate sale prices. The company wants to deploy the model to make predictions without managing servers or infrastructure. Which solution meets these requirements?",
        options: [
            "A. Deploy the model on an Amazon EC2 instance.",
            "B. Deploy the model on an Amazon Elastic Kubernetes Service (Amazon EKS) cluster.",
            "C. Deploy the model by using Amazon CloudFront with an Amazon S3 integration.",
            "D. Deploy the model by using an Amazon SageMaker endpoint."
        ],
        rightAnswer: "D. Deploy the model by using an Amazon SageMaker endpoint."
    },
    {
        question: "A company wants to develop an AI application to help its employees check open customer claims, identify details for a specific claim, and access documents for a claim. Which solution meets these requirements?",
        options: [
            "A. Use Agents for Amazon Bedrock with Amazon Fraud Detector to build the application.",
            "B. Use Agents for Amazon Bedrock with Amazon Bedrock knowledge bases to build the application.",
            "C. Use Amazon Personalize with Amazon Bedrock knowledge bases to build the application.",
            "D. Use Amazon SageMaker to build the application by training a new ML model."
        ],
        rightAnswer: "B. Use Agents for Amazon Bedrock with Amazon Bedrock knowledge bases to build the application."
    },
    {
        question: "A manufacturing company uses AI to inspect products and find any damages or defects. Which type of AI application is the company using?",
        options: [
            "A. Recommendation system",
            "B. Natural language processing (NLP)",
            "C. Computer vision",
            "D. Image processing"
        ],
        rightAnswer: "C. Computer vision"
    },
    {
        question: "A company wants to create an ML model to predict customer satisfaction. The company needs fully automated model tuning. Which AWS service meets these requirements?",
        options: [
            "A. Amazon Personalize",
            "B. Amazon SageMaker",
            "C. Amazon Athena",
            "D. Amazon Comprehend"
        ],
        rightAnswer: "B. Amazon SageMaker"
    },
    {
        question: "A bank has fine-tuned a large language model (LLM) to expedite the loan approval process. During an external audit of the model, the company discovered that the model was approving loans at a faster pace for a specific demographic than for other demographics. How should the bank fix this issue MOST cost-effectively?",
        options: [
            "A. Include more diverse training data. Fine-tune the model again by using the new data.",
            "B. Use Retrieval Augmented Generation (RAG) with the fine-tuned model.",
            "C. Use AWS Trusted Advisor checks to eliminate bias.",
            "D. Pre-train a new LLM with more diverse training data."
        ],
        rightAnswer: "A. Include more diverse training data. Fine-tune the model again by using the new data."
    },
    {
        question: "A company needs to log all requests made to its Amazon Bedrock API. The company must retain the logs securely for 5 years at the lowest possible cost. Which combination of AWS service and storage class meets these requirements? (Choose two.)",
        options: [
            "A. AWS CloudTrail",
            "B. Amazon CloudWatch",
            "C. AWS Audit Manager",
            "D. Amazon S3 Intelligent-Tiering",
            "E. Amazon S3 Standard"
        ],
        rightAnswer: ["A. AWS CloudTrail", "D. Amazon S3 Intelligent-Tiering"]
    },
    {
        question: "An ecommerce company wants to improve search engine recommendations by customizing the results for each user of the company's ecommerce platform. Which AWS service meets these requirements?",
        options: [
            "A. Amazon Personalize",
            "B. Amazon Kendra",
            "C. Amazon Rekognition",
            "D. Amazon Transcribe"
        ],
        rightAnswer: "A. Amazon Personalize"
    },
    {
        question: "A hospital is developing an AI system to assist doctors in diagnosing diseases based on patient records and medical images. To comply with regulations, the sensitive patient data must not leave the country the data is located in. Which data governance strategy will ensure compliance and protect patient privacy?",
        options: [
            "A. Data residency",
            "B. Data quality",
            "C. Data discoverability",
            "D. Data enrichment"
        ],
        rightAnswer: "A. Data residency"
    },
    {
        question: "A company needs to monitor the performance of its ML systems by using a highly scalable AWS service. Which AWS service meets these requirements?",
        options: [
            "A. Amazon CloudWatch",
            "B. AWS CloudTrail",
            "C. AWS Trusted Advisor",
            "D. AWS Config"
        ],
        rightAnswer: "A. Amazon CloudWatch"
    },
    {
        question: "An AI practitioner is developing a prompt for an Amazon Titan model. The model is hosted on Amazon Bedrock. The AI practitioner is using the model to solve numerical reasoning challenges. The AI practitioner adds the following phrase to the end of the prompt: 'Ask the model to show its work by explaining its reasoning step by step.' Which prompt engineering technique is the AI practitioner using?",
        options: [
            "A. Chain-of-thought prompting",
            "B. Prompt injection",
            "C. Few-shot prompting",
            "D. Prompt templating"
        ],
        rightAnswer: "A. Chain-of-thought prompting"
    },
    {
        question: "Which AWS service makes foundation models (FMs) available to help users build and scale generative AI applications?",
        options: [
            "A. Amazon Q Developer",
            "B. Amazon Bedrock",
            "C. Amazon Kendra",
            "D. Amazon Comprehend"
        ],
        rightAnswer: "B. Amazon Bedrock"
    },
    {
        question: "A company is building a mobile app for users who have a visual impairment. The app must be able to hear what users say and provide voice responses. Which solution will meet these requirements?",
        options: [
            "A. Use a deep learning neural network to perform speech recognition.",
            "B. Build ML models to search for patterns in numeric data.",
            "C. Use generative AI summarization to generate human-like text.",
            "D. Build custom models for image classification and recognition."
        ],
        rightAnswer: "A. Use a deep learning neural network to perform speech recognition."
    },
    {
        question: "A company wants to enhance response quality for a large language model (LLM) for complex problem-solving tasks. The tasks require detailed reasoning and a step-by-step explanation process. Which prompt engineering technique meets these requirements?",
        options: [
            "A. Few-shot prompting",
            "B. Zero-shot prompting",
            "C. Directional stimulus prompting",
            "D. Chain-of-thought prompting"
        ],
        rightAnswer: "D. Chain-of-thought prompting"
    },
    {
        question: "A company wants to keep its foundation model (FM) relevant by using the most recent data. The company wants to implement a model training strategy that includes regular updates to the FM. Which solution meets these requirements?",
        options: [
            "A. Batch learning",
            "B. Continuous pre-training",
            "C. Static training",
            "D. Latent training"
        ],
        rightAnswer: "B. Continuous pre-training"
    },
    {
        question: "Which option is a characteristic of AI governance frameworks for building trust and deploying human-centered AI technologies?",
        options: [
            "A. Expanding initiatives across business units to create long-term business value",
            "B. Ensuring alignment with business standards, revenue goals, and stakeholder expectations",
            "C. Overcoming challenges to drive business transformation and growth",
            "D. Developing policies and guidelines for data, transparency, responsible AI, and compliance"
        ],
        rightAnswer: "D. Developing policies and guidelines for data, transparency, responsible AI, and compliance"
    },
    {
        question: "An ecommerce company is using a generative AI chatbot to respond to customer inquiries. The company wants to measure the financial effect of the chatbot on the company's operations. Which metric should the company use?",
        options: [
            "A. Number of customer inquiries handled",
            "B. Cost of training AI models",
            "C. Cost for each customer conversation",
            "D. Average handled time (AHT)"
        ],
        rightAnswer: "C. Cost for each customer conversation"
    },
    {
        question: "A company wants to find groups for its customers based on the customers' demographics and buying patterns. Which algorithm should the company use to meet this requirement?",
        options: [
            "A. K-nearest neighbors (k-NN)",
            "B. K-means",
            "C. Decision tree",
            "D. Support vector machine"
        ],
        rightAnswer: "B. K-means"
    },
    {
        question: "A company's large language model (LLM) is experiencing hallucinations. How can the company decrease hallucinations?",
        options: [
            "A. Set up Agents for Amazon Bedrock to supervise the model training.",
            "B. Use data pre-processing and remove any data that causes hallucinations.",
            "C. Decrease the temperature inference parameter for the model.",
            "D. Use a foundation model (FM) that is trained to not hallucinate."
        ],
        rightAnswer: "C. Decrease the temperature inference parameter for the model."
    },
    {
        question: "A company is using a large language model (LLM) on Amazon Bedrock to build a chatbot. The chatbot processes customer support requests. To resolve a request, the customer and the chatbot must interact a few times. Which solution gives the LLM the ability to use content from previous customer messages?",
        options: [
            "A. Turn on model invocation logging to collect messages.",
            "B. Add messages to the model prompt.",
            "C. Use Amazon Personalize to save conversation history.",
            "D. Use Provisioned Throughput for the LLM."
        ],
        rightAnswer: "B. Add messages to the model prompt."
    },
    {
        question: "A company's employees provide product descriptions and recommendations to customers when customers call the customer service center. These recommendations are based on where the customers are located. The company wants to use foundation models (FMs) to automate this process. Which AWS service meets these requirements?",
        options: [
            "A. Amazon Macie",
            "B. Amazon Transcribe",
            "C. Amazon Bedrock",
            "D. Amazon Textract"
        ],
        rightAnswer: "C. Amazon Bedrock"
    },
    {
        question: "A company wants to upload customer service email messages to Amazon S3 to develop a business analysis application. The messages sometimes contain sensitive data. The company wants to receive an alert every time sensitive information is found. Which solution fully automates the sensitive information detection process with the LEAST development effort?",
        options: [
            "A. Configure Amazon Macie to detect sensitive information in the documents that are uploaded to Amazon S3.",
            "B. Use Amazon SageMaker endpoints to deploy a large language model (LLM) to redact sensitive data.",
            "C. Develop multiple regex patterns to detect sensitive data. Expose the regex patterns on an Amazon SageMaker notebook.",
            "D. Ask the customers to avoid sharing sensitive information in their email messages."
        ],
        rightAnswer: "A. Configure Amazon Macie to detect sensitive information in the documents that are uploaded to Amazon S3."
    },
    {
        question: "A company wants to deploy some of its resources in the AWS Cloud. To meet regulatory requirements, the data must remain local and on premises. There must be low latency between AWS and the company resources. Which AWS service or feature can be used to meet these requirements?",
        options: [
            "A. AWS Local Zones",
            "B. Availability Zones",
            "C. AWS Outposts",
            "D. AWS Wavelength Zones"
        ],
        rightAnswer: "C. AWS Outposts"
    },
    {
        question: "Which option is a benefit of using Amazon SageMaker Model Cards to document AI models?",
        options: [
            "A. Providing a visually appealing summary of a mode's capabilities.",
            "B. Standardizing information about a model's purpose, performance, and limitations.",
            "C. Reducing the overall computational requirements of a model.",
            "D. Physically storing models for archival purposes."
        ],
        rightAnswer: "B. Standardizing information about a model's purpose, performance, and limitations."
    },
    {
        question: "What does an F1 score measure in the context of foundation model (FM) performance?",
        options: [
            "A. Model precision and recall",
            "B. Model speed in generating responses",
            "C. Financial cost of operating the model",
            "D. Energy efficiency of the model's computations"
        ],
        rightAnswer: "A. Model precision and recall"
    },
    {
        question: "A company deployed an AI/ML solution to help customer service agents respond to frequently asked questions. The questions can change over time. The company wants to give customer service agents the ability to ask questions and receive automatically generated answers to common customer questions. Which strategy will meet these requirements MOST cost-effectively?",
        options: [
            "A. Fine-tune the model regularly.",
            "B. Train the model by using context data.",
            "C. Pre-train and benchmark the model by using context data.",
            "D. Use Retrieval Augmented Generation (RAG) with prompt engineering techniques."
        ],
        rightAnswer: "D. Use Retrieval Augmented Generation (RAG) with prompt engineering techniques."
    },
    {
        question: "A company built an AI-powered resume screening system. The company used a large dataset to train the model. The dataset contained resumes that were not representative of all demographics. Which core dimension of responsible AI does this scenario present?",
        options: [
            "A. Fairness",
            "B. Explainability",
            "C. Privacy and security",
            "D. Transparency"
        ],
        rightAnswer: "A. Fairness"
    },
    {
        question: "A global financial company has developed an ML application to analyze stock market data and provide stock market trends. The company wants to continuously monitor the application development phases and to ensure that company policies and industry regulations are followed. Which AWS services will help the company assess compliance requirements? (Choose two.)",
        options: [
            "A. AWS Audit Manager",
            "B. AWS Config",
            "C. Amazon Inspector",
            "D. Amazon CloudWatch",
            "E. AWS CloudTrail"
        ],
        rightAnswer: ["A. AWS Audit Manager", "B. AWS Config"]
    },
    {
        question: "A company wants to improve the accuracy of the responses from a generative AI application. The application uses a foundation model (FM) on Amazon Bedrock. Which solution meets these requirements MOST cost-effectively?",
        options: [
            "A. Fine-tune the FM.",
            "B. Retrain the FM.",
            "C. Train a new FM.",
            "D. Use prompt engineering."
        ],
        rightAnswer: "D. Use prompt engineering."
    },
    {
        question: "A company wants to identify harmful language in the comments section of social media posts by using an ML model. The company will not use labeled data to train the model. Which strategy should the company use to identify harmful language?",
        options: [
            "A. Use Amazon Rekognition moderation.",
            "B. Use Amazon Comprehend toxicity detection.",
            "C. Use Amazon SageMaker built-in algorithms to train the model.",
            "D. Use Amazon Polly to monitor comments."
        ],
        rightAnswer: "B. Use Amazon Comprehend toxicity detection."
    },
    {
        question: "A media company wants to analyze viewer behavior and demographics to recommend personalized content. The company wants to deploy a customized ML model in its production environment. The company also wants to observe if the model quality drifts over time. Which AWS service or feature meets these requirements?",
        options: [
            "A. Amazon Rekognition",
            "B. Amazon SageMaker Clarify",
            "C. Amazon Comprehend",
            "D. Amazon SageMaker Model Monitor"
        ],
        rightAnswer: "D. Amazon SageMaker Model Monitor"
    },
    {
        question: "A manufacturing company wants to create product descriptions in multiple languages. Which AWS service will automate this task?",
        options: [
            "A. Amazon Translate",
            "B. Amazon Transcribe",
            "C. Amazon Kendra",
            "D. Amazon Polly"
        ],
        rightAnswer: "A. Amazon Translate"
    }
];