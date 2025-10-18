import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms


class FeatureDiscriminator(nn.Module):
    """
    Discriminator that works on feature representations instead of raw images.
    Useful for discriminating between DINOv2 features from real vs adapted images.
    """
    
    def __init__(self, input_dim=768, hidden_dims=[512, 256, 128]):
        super(FeatureDiscriminator, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        
        # Build layers dynamically
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LeakyReLU(0.2, inplace=False),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        # Final classification layer
        layers.extend([
            nn.Linear(prev_dim, 1),
            nn.Sigmoid()
        ])
        
        self.discriminator = nn.Sequential(*layers)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using normal distribution"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through feature discriminator.
        
        Args:
            x: Input feature tensor of shape (B, feature_dim)
            
        Returns:
            Probability that input features are from real image (0-1)
        """
        return self.discriminator(x)


class GANLoss:
    """
    Helper class for GAN losses
    """
    
    @staticmethod
    def adversarial_loss(discriminator_output, target_is_real):
        """
        Standard GAN loss for discriminator and generator.
        
        Args:
            discriminator_output: Output from discriminator (probabilities)
            target_is_real: Whether the target should be classified as real
            
        Returns:
            Loss value
        """
        if target_is_real:
            target = torch.ones_like(discriminator_output)
        else:
            target = torch.zeros_like(discriminator_output)
        
        return F.binary_cross_entropy(discriminator_output, target)
    
    @staticmethod
    def generator_loss(discriminator_output_on_fake):
        """
        Generator loss - wants discriminator to classify fake as real.
        
        Args:
            discriminator_output_on_fake: Discriminator output on generated data
            
        Returns:
            Generator loss
        """
        target = torch.ones_like(discriminator_output_on_fake)
        return F.binary_cross_entropy(discriminator_output_on_fake, target)
    
    @staticmethod
    def discriminator_loss(real_output, fake_output):
        """
        Discriminator loss - wants to classify real as real, fake as fake.
        
        Args:
            real_output: Discriminator output on real data
            fake_output: Discriminator output on fake data
            
        Returns:
            Discriminator loss
        """
        real_loss = F.binary_cross_entropy(real_output, torch.ones_like(real_output))
        fake_loss = F.binary_cross_entropy(fake_output, torch.zeros_like(fake_output))
        return (real_loss + fake_loss) / 2

def preprocess_image(image_path, target_size=(224, 224)):
    """Load and preprocess image for discriminator input"""
    img = Image.open(image_path).convert('RGB')
    
    # Resize to discriminator input size
    img = img.resize(target_size)
    
    # Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    return img_tensor, img

def test_discriminators():
    """Test both discriminator architectures"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    real = '/fs/nexus-scratch/aliu1237/carla_splits/left/CAM_F/routes_10mshortroutes_Town01_Scenario9_route0_11_23_20_36_48__0069.png'
    virtual = '/fs/nexus-scratch/aliu1237/carla_splits/left/CAM_F/routes_10mshortroutes_Town01_Scenario9_route0_11_23_20_36_48__0070.png'
    navsim_tensor, navsim_img = preprocess_image(real)
    carla_tensor, carla_img = preprocess_image(virtual)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    navsim_tensor = navsim_tensor.to(device)
    carla_tensor = carla_tensor.to(device)

    # Test ImageDiscriminator
    print("Testing ImageDiscriminator...")
    image_discriminator = ImageDiscriminator().to(device)
    batch_images = torch.cat([navsim_tensor, carla_tensor], dim=0)  # Combine real and virtual images
    image_output = image_discriminator(batch_images)
    print(f"Image discriminator output shape: {image_output.shape}")
    print(f"Image discriminator output: {image_output.squeeze()}")
    
    # Test FeatureDiscriminator
    print("\nTesting FeatureDiscriminator...")
    feature_discriminator = FeatureDiscriminator(input_dim=768).to(device)
    batch_features = torch.randn(4, 768).to(device)  # 4 feature vectors
    feature_output = feature_discriminator(batch_features)
    print(f"Feature discriminator output shape: {feature_output.shape}")
    print(f"Feature discriminator output: {feature_output.squeeze()}")
    
    # Test loss functions
    print("\nTesting loss functions...")
    real_pred = image_output[0:1]  # First output (real image)
    fake_pred = image_output[1:2]  # Second output (virtual image)
    
    d_loss = GANLoss.discriminator_loss(real_pred, fake_pred)
    g_loss = GANLoss.generator_loss(fake_pred)
    
    print(f"Discriminator loss: {d_loss.item():.4f}")
    print(f"Generator loss: {g_loss.item():.4f}")

    # Interpretation
    print(f"\nInterpretation:")
    print(f"NavSim classified as: {'Real' if real_pred.item() > 0.5 else 'Fake'}")
    print(f"CARLA classified as: {'Real' if fake_pred.item() > 0.5 else 'Fake'}")


if __name__ == "__main__":
    test_discriminators()