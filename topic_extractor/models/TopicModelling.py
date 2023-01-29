from typing import List, Union
import pickle as pkl
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from transformers import pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA

class TopicModelling(object):
    """A Multimodal deep learning model for topic modelling on both texts and images.
    By default, the model uses Sequence Transformers for text and image embeddings, BERTopic for topic modelling and VIT-GPT2 for image captioning.
    Additionally, it also needs a dimensionality reduction model to unify the dimension shapes of the text and image embeddings.
    Training the model requires 3 sets of data: the images, the image captions, the documents.
    If there's no image captions provided, the model will attempt to do image captioning automatically.
    Also note that since the model depends on Sequence Transformers, input strings from short to medium length (sentence -> paragraph) will give
    better results. Large documents should be splitted into paragraphs before they get fed to this model.
    """
    def __init__(self, img_emb_model=None,
                 txt_emb_model=None,
                 dimension_reduce=None,
                 img_caption_model=None,
                 topic_model=None,
                 device=-1, n_components=384, n_gram_range=(1,3), diversity=0.5) -> None:
        """Initialize the TopicModelling object.

        Parameters
        ----------
        img_emb_model : None, optional
            The sequence transformer to extract image features, by default SentenceTransformer
        txt_emb_model : None, optional
            The sequence transformer to extract text features, by default SentenceTransformer
        dimension_reduce : None, optional
            The dimensionality reduction model, by default PCA
        img_caption_model : None, optional
            Model for auto-generated image captions, by default vit-gpt2-image-captioning
        topic_model : None, optional
            Core model for topic modelling, by default it is BERTopic
        device : int, optional
            device id that specifies whether the GPU is being used, by default -1 means using CPU
        n_components : int, optional
            The new length that will be applied to the image embeddings, by default 384.
            If the dimension reduction model is None and n_components is -1, no dimensionality reduction will be applied
            during feature extractions.
        n_gram_range : tuple, optional
            Range of the N-Gram that BERTopic uses, by default (1,3)
        diversity : float, optional
            Specifies how diverse the resulted topics will be, 0 means no diversity and vice versa, by default 0.5
        """
        self.device = device
        # Set image embedding model
        if img_emb_model is not None:
            self._img_emb_model = img_emb_model
        else:
            self._img_emb_model = SentenceTransformer('clip-ViT-B-32')
        # Set text embedding model
        if txt_emb_model is not None:
            self._txt_emb_model = txt_emb_model
        else:
            self._txt_emb_model = SentenceTransformer("all-MiniLM-L6-v2")
        # Set dimensionality reduction model
        if dimension_reduce is not None:
            self._dimension_reduce = dimension_reduce
        elif n_components == -1:
            self._dimension_reduce = None
        else:
            self._dimension_reduce = PCA(n_components=n_components)
        # Set image captioning model
        if img_caption_model is not None:
            self._img_caption_model = img_caption_model
        else:
            self._img_caption_model = pipeline("image-to-text",
                                               model="nlpconnect/vit-gpt2-image-captioning",
                                               device=self.device)
        # Set topic model
        if topic_model is not None:
            self._topic_model = topic_model
        else:
            vectorizer_model = CountVectorizer(stop_words="english")
            self._topic_model = BERTopic(calculate_probabilities=False,
                                        low_memory=True,
                                        n_gram_range=n_gram_range,
                                        diversity=diversity,
                                        vectorizer_model=vectorizer_model)
        self._embeddings = None      
        
    @property
    def feature_embeddings(self):
        """Get the embedded feature matrix extracted by the model. The features can consist of both images and texts.
        """
        return self._embeddings
    
    @property
    def topic_model(self):
        """Get the topic model wrapped by this object.
        """
        return self._topic_model

    def _extract_features(self, images: List[str], docs: List[str], batch_size=32, is_training=False):
        """Extract the embedding features of images and documents using the Sequence Transformers.
        Also train the dimensionality reduction model if is_training is True.

        Parameters
        ----------
        images : List[str]
            List of paths or URL to the images in local or remote storage.
        docs : List[str]
            List of documents. Each document should be a sentence or a paragraph only.
        batch_size : int, optional
            Size of the batch to iterate the image set, by default 32
        is_training : bool, optional
            Toggle training mode of this function, by default False
        
        Returns
        ----------
        A NumPy array of the embedding matrix of both images and documents
        """
        # Prepare images
        nr_iterations = int(np.ceil(len(images) / batch_size))
        
        # Embed images per batch
        img_embeddings = []
        for i in tqdm(range(nr_iterations)):
            start_index = i * batch_size
            end_index = (i * batch_size) + batch_size

            images_to_embed = [Image.open(filepath) for filepath in images[start_index:end_index]]
            
            img_emb = self._img_emb_model.encode(images_to_embed, show_progress_bar=False)
            img_embeddings.extend(img_emb.tolist())

            # Close images
            for image in images_to_embed:
                image.close()       
        img_embeddings = np.array(img_embeddings)
        
        # Dimensionality reduction on image features
        if self._dimension_reduce is not None:
            if is_training:
                img_embeddings = self._dimension_reduce.fit_transform(img_embeddings)
            else:
                img_embeddings = self._dimension_reduce.transform(img_embeddings)
        
        # Prepare documents
        text_embeddings = self._txt_emb_model.encode(docs, show_progress_bar=True)
        text_embeddings = np.array(text_embeddings)
        
        emb = np.concatenate([img_embeddings, text_embeddings])
        if is_training:
            self._embeddings = emb
        return emb
        
    def _caption_images(self, images: List[str]) -> List[str]:
        """Call the image captioning model to generate captions for the input images.

        Parameters
        ----------
        images : List[str]
            List of paths or URL to the images in local or remote storage.

        Returns
        -------
        List[str]
            List of image captions
        """
        captions = self._img_caption_model(images)
        captions = [result[0]["generated_text"] for result in captions]
        return captions

    def fit(self, images: List[str], image_captions: Union[List[str], None], docs: List[str], batch_size=32):
        """Fit the Topic Model with the input data: images, captions and documents.
        Calling this function will change the inner feature embeddings of this current model.

        Parameters
        ----------
        images : List[str]
            List of paths or URL to the images in local or remote storage.
        image_captions : Union[List[str], None]
            List of image captions. If None, the model will automatically generate image captions.
        docs : List[str]
            List of documents. Each document should be a sentence or a paragraph only.
        batch_size : int, optional
            Size of the batch to iterate the image set, by default 32
        """
        self._extract_features(images, docs, batch_size, True)
        if image_captions is None:
            image_captions = self._caption_images(images)
        all_docs = image_captions + docs
        self._topic_model = self._topic_model.fit(all_docs, self._embeddings)
        
    def transform(self, images: List[str], image_captions: Union[List[str], None], docs: List[str], batch_size=32, use_old_embeddings=False):
        """Model the input data with the topics this model has learned.

        Parameters
        ----------
        images : List[str]
            List of paths or URL to the images in local or remote storage.
        image_captions : Union[List[str], None]
            List of image captions. If None, the model will automatically generate image captions.
        docs : List[str]
            List of documents. Each document should be a sentence or a paragraph only.
        batch_size : int, optional
            Size of the batch to iterate the image set, by default 32
        use_old_embeddings : bool, optional
            Toggle whether or not to reuse the embeddings created by the fit function, by default False

        Returns
        -------
        The return object is according to the transform function of the Topic Model that is used as the core
        of this model. For example, BERTopic will return a tuple
        consists of a list of topic ids corresponding to each input and the probabilities of those topic predictions
        """
        emb = self._embeddings
        if not use_old_embeddings:
            emb = self._extract_features(images, docs, batch_size, False)
        if image_captions is None:
            image_captions = self._caption_images(images)
        all_docs = image_captions + docs
        return self._topic_model.transform(all_docs, emb)
    
    def fit_transform(self, images: List[str], image_captions: Union[List[str], None], docs: List[str], batch_size=32, use_old_embeddings=False):
        """Fit the model and transform on the input data at the same time.

        Parameters
        ----------
        Same as 'transform' function

        Returns
        -------
        Same as 'transform' function
        """
        self.fit(images, image_captions, docs, batch_size)
        return self.transform(images, image_captions, docs, batch_size, use_old_embeddings)
    
    def get_n_top_topics(self, transform_result, n=10):
        """A helper function to parse the result of transform in order to get the
        top N most frequent topics appear in the result. The function only works for
        when BERTopic is used as the core topic model.

        Parameters
        ----------
        transform_result :
            The result of the transform function, based on BERTopic interface
        n : int, optional
            How many most frequent topics to get, by default 10

        Returns
        -------
        List[str]
            List of n most frequent topics in the data
        """
        topics = transform_result[0]
        topic_labels = self._topic_model.generate_topic_labels(nr_words=3, topic_prefix=False, separator=" ")
        df = pd.DataFrame({"Topic": [topic_labels[t + 1] for t in topics]})
        df["Topic"] = df[df.Topic != topic_labels[0]]
        df = df.groupby(["Topic"], sort=False)["Topic"].agg(Count="count").sort_values("Count", ascending=False).reset_index()
        return df[0:n]
    
    def save(self, topic_model_path: str, dim_reduce_path: Union[str, None]=None):
        """Save the model to files. More specifically, this function only saves
        the dimensionality reduction model and the core topic model, since the other
        models are pretrained models from HuggingFace so there's no need to save.

        Parameters
        ----------
        topic_model_path : str
            Path to save the topic model
        dim_reduce_path : str
            Path to save the dimensionality reduction model
        """
        if self._dimension_reduce is not None and dim_reduce_path is not None:
            with open(dim_reduce_path,"wb") as pklFile:
                pkl.dump(self._dimension_reduce, pklFile)
        self._topic_model.save(topic_model_path)
        
    @classmethod
    def load(cls, topic_model_path: str,
                dim_reduce_path: Union[str, None]=None,
                img_emb_model=None,
                txt_emb_model=None,
                img_caption_model=None,
                device=-1):
        """Load a topic model with BERTopic backbone and a dimensionality reduction model.
        The other component models are pretrained models that can be loaded by other packages and passed to this function.
        Check the documentation of the constructor of this class for more details about other parameters.

        Parameters
        ----------
        topic_model_path : str
            Path to save the topic model
        dim_reduce_path : Union[str, None], optional
            Path to save the dimensionality reduction model, by default None

        Returns
        -------
        TopicModelling
            A new instance of TopicModelling
        """
        bertopic = BERTopic.load(topic_model_path)
        dim_reduce = None
        
        if dim_reduce_path is not None:
            with open(dim_reduce_path,"rb") as pklFile:
                dim_reduce = pkl.load(pklFile)
        
        return TopicModelling(
            img_emb_model,
            txt_emb_model,
            dim_reduce,
            img_caption_model,
            bertopic,
            device
        )