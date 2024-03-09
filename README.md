## VODKA (Voting Over Distilled Knowledge Associations)

Vodka is image segmentation pipeline designed to automatically segment images based on a user prompt. The pipeline currently uses 3 models:
- [SAM (Segment Anything Model)](https://segment-anything.com/)
- [BLIP (image captioning model)](https://huggingface.co/docs/transformers/en/model_doc/blip)
- [UAE-Large-V1 (embedding model)](https://huggingface.co/WhereIsAI/UAE-Large-V1)

## Components

- **`app.py`**:  Is a Streamlit  application that integrates various components of the project together. It processes the top-k cutouts from SAM and the user prompt to generate the best matching image segmentation.

- **`image_to_masks.py`**: Responsible for converting input images into cutouts using SAM. This helps in identifying distinct objects within the images to be processed further.

- **`masks_to_text.py`**: Generates textual descriptions for the SAM cutouts. This description is then used to compute similarities with the users prompt to identify the right cutout.

- **`text_to_embeddings.py`**: Transforms textual descriptions into embeddings. These embeddings are used to calculate cosine similarity with the users prompt to select the right cutout.  

## How to run

Follow these steps to set up and run the project: 

### Step 1: Install Required Libraries 
```bash
  pip install -r requirements.txt
```
  
### Step 2: Download and Place Models
You will need to download specific models and place them in the `models` directory as follows:
  ```
  models
	├── BLIP
	├── SAM
	└── UAE-Large-V1
  ```
 - **BLIP** and **UAE-Large-V1** models are available on Hugging Face. Make sure you have `git-lfs` installed before cloning (`git lfs install`):
	
 	```bash
	git clone https://huggingface.co/WhereIsAI/UAE-Large-V1
	git clone https://huggingface.co/Salesforce/blip-image-captioning-large
	```
- **SAM** (`sam_vit_h_4b8939`) can be downloaded from the link below. After downloading, place the checkpoint in a `SAM` folder under `models`.
	- [Model checkpoint](https://github.com/facebookresearch/segment-anything#model-checkpoints)

### Step 3: Generate Top-k Segments
Use the `image_to_masks.py` file to generate the top-k segments from an input image.

### Step 4: Obtain Appropriate Segments Based on the Prompt
After you have the top-k cutouts, you can use the Streamlit app or the Python scripts `masks_to_text.py` and `text_to_embeddings.py` to get the appropriate segments based on the prompt. To run the Streamlit app:
```bash
  streamlit run app.py
```

### Tips for Best Results
For optimal results, use a descriptive prompt that names the object and mentions the color of the object.\
A sample image is available in the `images` folder.



	



