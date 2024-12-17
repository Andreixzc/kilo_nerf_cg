import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os

# Import the model class from the training script
from train_kilonerf import KiloNerf, render_rays

@torch.no_grad()
def test(model, hn, hf, dataset, chunk_size=5, img_index=0, nb_bins=192, H=400, W=400, device='cuda'):
    ray_origins = dataset[img_index * H * W: (img_index + 1) * H * W, :3]
    ray_directions = dataset[img_index * H * W: (img_index + 1) * H * W, 3:6]

    data = []
    for i in range(int(np.ceil(H / chunk_size))):
        ray_origins_ = ray_origins[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)
        ray_directions_ = ray_directions[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)
        regenerated_px_values = render_rays(model, ray_origins_, ray_directions_, hn=hn, hf=hf, nb_bins=nb_bins)
        data.append(regenerated_px_values)
    img = torch.cat(data).data.cpu().numpy().reshape(H, W, 3)

    plt.figure()
    plt.imshow(img)
    plt.axis('off')
    plt.savefig(f'novel_views/img_{img_index}.png', bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    # CUDA setup
    print("\n=== CUDA Setup ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    
    # Load the saved checkpoint
    print("\n=== Loading Model ===")
    checkpoint = torch.load('kilonerf_checkpoint.pth')
    
    # Create model instance with same parameters
    model = KiloNerf(
        N=checkpoint['N'],
        embedding_dim_pos=checkpoint['embedding_dim_pos'],
        embedding_dim_direction=checkpoint['embedding_dim_direction'],
        scene_scale=checkpoint['scene_scale']
    ).to(device)
    
    # Load the saved model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set to evaluation mode
    
    # Load testing dataset
    print("\n=== Loading Testing Data ===")
    testing_dataset = torch.from_numpy(np.load('testing_data.pkl', allow_pickle=True))
    print(f"Testing dataset size: {testing_dataset.shape}")
    
    # Create output directory
    os.makedirs('novel_views', exist_ok=True)
    
    # Generate test images
    print("\n=== Generating Test Images ===")
    num_test_images = 200  # You can modify this
    for idx in range(num_test_images):
        if (idx + 1) % 10 == 0:
            print(f"Generating image {idx + 1}/{num_test_images}")
        test(model, hn=2, hf=6, dataset=testing_dataset, img_index=idx, 
             nb_bins=192, H=400, W=400, device=device)
    
    print("\nTesting completed! Images saved in 'novel_views' directory.")