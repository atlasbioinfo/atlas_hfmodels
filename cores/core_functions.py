import logging, argparse,os,sys


def login_to_hub():
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        user_info = api.whoami()
        print("\nSuccessfully logged in to Hugging Face Hub!")
        print(f"Welcome ({user_info['name']})")

        print("\n=== User Information ===")
        print(f"Username: {user_info['name']}")
        print(f"Full Name: {user_info['fullname']}")
        print(f"==========================")
        print("\n")
    except Exception:
        # Not logged in, prompt for login
        print("Please login to Hugging Face Hub first.")
        print("You can get your access token from https://huggingface.co/settings/tokens")
        token = input("Enter your Hugging Face access token: ").strip()
        
        # Login with provided token
        from huggingface_hub import login
        try:
            login(token=token)
            print("Successfully logged in to Hugging Face Hub!")
            from huggingface_hub import HfApi
            api = HfApi()
            user_info = api.whoami()
        except Exception as e:
            print(f"Login failed: {str(e)}")
            sys.exit(1)
    
    atlas_path = os.path.expanduser("~/.atlas")
    if not os.path.exists(atlas_path):
        os.makedirs(atlas_path)

    username_file = os.path.join(atlas_path, "hf_username.info")
    
    with open(username_file, "w") as f:
        f.write(user_info['name'])
    

def get_username():
    """
    Get the Hugging Face username from the stored info file
    
    Returns:
        str: The username stored in ~/.atlas/hf_username.info
    """
    atlas_path = os.path.expanduser("~/.atlas")
    username_file = os.path.join(atlas_path, "hf_username.info")
    
    if not os.path.exists(username_file):
        logging.error("Username info not found. Please login first.")
        sys.exit(1)
        
    with open(username_file, "r") as f:
        username = f.read().strip()
        
    return username

def remove_models(repo_name, force=False):
    from huggingface_hub import HfApi
    api = HfApi()
    try:
        api.repo_info(repo_name)
        if not force:
            confirm = input(f"Are you sure you want to delete repository {repo_name}? [y/N]: ").lower()
            if confirm != 'y':
                logging.info("Deletion cancelled")
                return
                
        api.delete_repo(repo_name)
        logging.info(f"Repository {repo_name} removed successfully")
        
    except Exception as e:
        if "404" in str(e):
            logging.error(f"Repository {repo_name} does not exist")
        else:
            logging.error(f"Failed to remove repository {repo_name}: {str(e)}")
        raise

def list_models(filter_keyword, username):
    from huggingface_hub import HfApi
    api = HfApi()
    models = api.list_models(author=username, filter=filter_keyword)
    for model in models:
        print(f"Model: {model.modelId}")
        # List files in the model repository
        try:
            files = api.list_repo_files(repo_id=model.modelId, repo_type="model")
            if files:
                print("  Files:")
                for file in files:
                    if file.endswith(".bin") or file.endswith(".safetensors"):
                        print(f"    - {file}")
        except Exception as e:
            print(f"  Error listing files: {str(e)}")
        print()

def upload_models(model_path, username,repo_name=None, public=False, ):
    """
    Upload multiple models from a directory to Hugging Face Model Hub
    
    Args:
        model_path (str): Path to the directory containing models
        repo_name (str, optional): Repository name for upload. Defaults to model folder name
        public (bool, optional): Whether to make the model public. Defaults to False
    """
    logging.basicConfig(level=logging.INFO)
    logging.info("Uploading models to Hugging Face Model Hub")
    
    model_dirs = [d for d in os.listdir(model_path) 
                    if os.path.isdir(os.path.join(model_path, d))]
        
    valid_model_dirs = []
    for d in model_dirs:
        model_path_dir = os.path.join(model_path, d)
        if os.path.exists(os.path.join(model_path_dir, "config.json")) and \
           (os.path.exists(os.path.join(model_path_dir, "pytorch_model.bin")) or \
            os.path.exists(os.path.join(model_path_dir, "model.safetensors"))):
            valid_model_dirs.append(d)
    
    model_count = len(valid_model_dirs)
    if model_count == 0:
        logging.error(f"No models found in {model_path}")
        return
    logging.info(f"Found {model_count} valid model directories in {model_path}:")
    for model_dir in valid_model_dirs:
        logging.info(f"  - {model_dir}")
        
    if repo_name is None:
        repo_name = os.path.basename(model_path)
        logging.info(f"Using model directory name as repository name: {repo_name}")
    else:
        logging.info(f"Using provided repository name: {repo_name}")
    

    from huggingface_hub import HfApi
    api = HfApi()
    
    try:
        api.repo_info(repo_id=f"{username}/{repo_name}", repo_type="model")
        logging.info(f"Repository {repo_name} exists, will not create a new one")
        files = api.list_repo_files(repo_id=f"{username}/{repo_name}", repo_type="model")
        if files:
            logging.info("Current files in repository:")
            for file in files:
                logging.info(f"  - {file}")
        else:
            logging.info("Repository is empty")
    except Exception:
        logging.info(f"Creating new repository: {repo_name}")
        api.create_repo(
            repo_id=repo_name,
            repo_type="model",
            private=not public
        )
        logging.info(f"Repository {repo_name} created successfully")
    # Upload all files from each valid model directory
    for model_dir in valid_model_dirs:
        model_dir_path = os.path.join(model_path, model_dir)
        logging.info(f"Uploading files from {model_dir_path}")
        
        try:
            # Get list of all files in the model directory
            model_files = [f for f in os.listdir(model_dir_path) 
                         if os.path.isfile(os.path.join(model_dir_path, f))]
            
            # Upload each file to the corresponding subfolder
            for file in model_files:
                file_path = os.path.join(model_dir_path, file)
                api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=f"{model_dir}/{file}",
                    repo_id=f"{username}/{repo_name}",
                    repo_type="model"
                )
                logging.info(f"  Uploaded {file} to {model_dir}/{file}")
                
            logging.info(f"Successfully uploaded all files for model {model_dir}")
            
        except Exception as e:
            logging.error(f"Failed to upload model {model_dir}: {str(e)}")
            # Continue with next model even if one fails
            continue
    
    logging.info(f"All models uploaded to repository {username}/{repo_name}")
    
    

def load_model_from_hub(repo_name, model_name):
    """
    Load a specific model from a Hugging Face Hub repository
    
    Args:
        repo_name (str): Repository name in format username/repo_name
        model_name (str): Name of the specific model (subfolder) to load
        
    Returns:
        tuple: (model, tokenizer) The loaded model and tokenizer
    """
    from transformers import AutoModel, AutoTokenizer
    
    full_model_path = f"{repo_name}/{model_name}"
    logging.info(f"Loading model from {full_model_path}")
    
    try:
        model = AutoModel.from_pretrained(full_model_path)
        tokenizer = AutoTokenizer.from_pretrained(full_model_path)
        return model, tokenizer
    except Exception as e:
        logging.error(f"Failed to load model: {str(e)}")
        raise

def download_dataset(repo_name, output_dir):
    """
    Download a dataset from Hugging Face Hub
    
    Args:
        repo_name (str): Repository name in format username/repo_name
        output_dir (str): Local directory to save the dataset
    """
    logging.info(f"Downloading dataset from {repo_name}")
    try:
        from datasets import load_dataset
        dataset = load_dataset(repo_name)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save dataset to disk
        output_path = os.path.join(output_dir, repo_name.split('/')[-1])
        dataset.save_to_disk(output_path)
        logging.info(f"Dataset successfully downloaded to {output_path}")
        
    except Exception as e:
        logging.error(f"Failed to download dataset: {str(e)}")
        raise

def check_dataset(repo_name):
    """
    Check dataset statistics from Hugging Face Hub
    """
    from huggingface_hub import HfApi
    api = HfApi()
    dataset_info = api.dataset_info(repo_name)
    print("\n=== Dataset Information ===")
    print(f"ID: {dataset_info.id}")
    print(f"Author: {dataset_info.author}")
    print(f"Created: {dataset_info.created_at}")
    print(f"Last Modified: {dataset_info.last_modified}")
    print(f"Private: {dataset_info.private}")
    print(f"Downloads: {dataset_info.downloads}")
    print(f"Likes: {dataset_info.likes}")
    print(f"Tags: {dataset_info.tags}")
    
    if dataset_info.card_data and dataset_info.card_data.get('dataset_info'):
        info = dataset_info.card_data['dataset_info']
        print("\n=== Dataset Statistics ===")
        print("Features:")
        for feature in info['features']:
            print(f"  - {feature['name']} ({feature['dtype']})")
        
        print("\nSplits:")
        for split in info['splits']:
            print(f"  - {split['name']}: {split['num_examples']} examples")
        
        print(f"\nDownload Size: {info['download_size']} bytes")
        print(f"Dataset Size: {info['dataset_size']} bytes")
    
    print("\n=== Files ===")
    for sibling in dataset_info.siblings:
        print(f"  - {sibling.rfilename}")
    print("=======================\n")
