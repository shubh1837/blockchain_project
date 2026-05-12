import os
import argparse
import torch
import torchvision
import torchxrayvision as xrv
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from src.dacnet import DenseNet121, FocalLoss

def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set up transforms
    transform = torchvision.transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224)
    ])

    # Initialize model first to get its pathologies
    print("Initializing DACNet DenseNet121...")
    model = DenseNet121(classCount=14, isTrained=False)
    
    # Load base weights if they exist to fine-tune from them
    weights_path = "data/dacnet.pth"
    if os.path.exists(weights_path):
        print(f"Loading base weights from {weights_path}")
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get('state_dict', checkpoint)
        
        # Strip module prefix just in case
        clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        model.densenet121.load_state_dict(clean_state_dict)

    model.pathologies = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
            'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

    model.train()
    model.to(device)

    # Load dataset
    print(f"Loading dataset from images: {args.imgpath}, CSV: {args.csvpath}")
    dataset = xrv.datasets.NIH_Dataset(imgpath=args.imgpath,
                                       csvpath=args.csvpath,
                                       transform=transform)

    # Align dataset labels with model outputs
    print("Aligning dataset labels with model outputs...")
    xrv.datasets.relabel_dataset(model.pathologies, dataset)

    # Filter out missing images
    print("Verifying image paths...")
    valid_indices = []
    # Fast path checking using a set of directory contents
    existing_files = set(os.listdir(args.imgpath))
    for i in tqdm(range(len(dataset.csv)), desc="Scanning files"):
        if dataset.csv.iloc[i]['Image Index'] in existing_files:
            valid_indices.append(i)
            
    print(f"Found {len(valid_indices)} valid images out of {len(dataset.csv)}.")
    
    # Use a subset if specified
    if args.subset > 0 and args.subset < len(valid_indices):
        print(f"Using a subset of {args.subset} images.")
        # Take the first N valid indices
        valid_indices = valid_indices[:args.subset]
    else:
        print(f"Using full valid dataset with {len(valid_indices)} images.")
        
    dataset = Subset(dataset, valid_indices)

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # Set up optimizer and loss function
    # Note: Use FocalLoss because DACNet requires it for class imbalance
    criterion = FocalLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Initialize GradScaler for mixed precision training (saves VRAM and speeds up training)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    # Ensure output dir exists
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    print("Starting training...")
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        # Use tqdm for progress bar
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            imgs = batch["img"].to(device)
            labels = batch["lab"].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with Automatic Mixed Precision (AMP)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(imgs)
                
                # Calculate loss only on valid labels (ignore NaNs)
                mask = ~torch.isnan(labels)
                
                if mask.sum() == 0:
                    continue # Skip if no valid labels
                
                loss = criterion(outputs[mask], labels[mask])
            
            # Backward pass with scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        print(f"Epoch {epoch+1} Average Loss: {epoch_loss / len(dataloader):.4f}")

    print(f"Saving model weights to {args.save_path}")
    torch.save(model.state_dict(), args.save_path)
    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain/Fine-tune TorchXRayVision model.")
    parser.add_argument("--imgpath", type=str, default="images", help="Path to images directory")
    parser.add_argument("--csvpath", type=str, default="Data_Entry_2017.csv", help="Path to CSV file")
    parser.add_argument("--save-path", type=str, default="data/fine_tuned_weights.pth", help="Path to save trained weights")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--subset", type=int, default=100, help="Number of images to use for training (0 for all)")
    
    args = parser.parse_args()
    main(args)
