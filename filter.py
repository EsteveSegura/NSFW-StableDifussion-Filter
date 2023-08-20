# Import necessary libraries
import torch
from torch import nn

# List of standard NSFW concepts to filter out
concepts = ['sexual', 'nude', 'sex', '18+', 'naked', 'nsfw', 'porn', 'dick', 'vagina',
            'naked child', 'explicit content', 'uncensored', 'fuck', 'nipples', 'visible nipples', 'naked breasts', 'areola']

# List of special concepts, focusing on protecting images of minors
special_concepts = ["little girl", "young child", "young girl"]

# Function to compute the cosine distance between image and text embeddings
def cosine_distance(image_embeds, text_embeds):
    # Normalize the image embeddings for better accuracy
    normalized_image_embeds = nn.functional.normalize(image_embeds)
    # Normalize the text embeddings for better accuracy
    normalized_text_embeds = nn.functional.normalize(text_embeds)
    # Compute the dot product between the normalized embeddings
    return torch.mm(normalized_image_embeds, normalized_text_embeds.t())

# Decorator to ensure no gradients are computed during the forward pass, for efficiency
@torch.no_grad()
def forward_inspect(self, clip_input, images):
    # Extract the output from the vision model, which gives embeddings for the images
    pooled_output = self.vision_model(clip_input)[1]
    # Project the embeddings to align with text embeddings
    image_embeds = self.visual_projection(pooled_output)

    # Calculate cosine distance between image embeddings and special concepts
    special_cos_dist = cosine_distance(
        image_embeds, self.special_care_embeds
    ).cpu().numpy()
    
    # Calculate cosine distance between image embeddings and standard NSFW concepts
    cos_dist = cosine_distance(image_embeds, self.concept_embeds).cpu().numpy()

    # Dictionary to store matched NSFW and special terms for each image
    matches = {"nsfw": [], "special": []}
    batch_size = image_embeds.shape[0]
    
    # Iterate over each image in the batch
    for i in range(batch_size):
        # Dictionary to store matching scores and concepts for the current image
        result_img = {
            "special_scores": {}, "special_care": [], "concept_scores": {}, "bad_concepts": []
        }

        adjustment = 0.0

        # Check each image against the list of special concepts
        for concet_idx in range(len(special_cos_dist[0])):
            concept_cos = special_cos_dist[i][concet_idx]
            concept_threshold = self.special_care_embeds_weights[concet_idx].item()
            
            # Compute how much the current image matches the special concept
            result_img["special_scores"][concet_idx] = round(
                concept_cos - concept_threshold + adjustment, 3
            )
            # If there's a strong match, flag the image and adjust the threshold for subsequent checks
            if result_img["special_scores"][concet_idx] > 0:
                result_img["special_care"].append(
                    {concet_idx, result_img["special_scores"][concet_idx]}
                )
                adjustment = 0.01
                matches["special"].append(special_concepts[concet_idx])

        # Check each image against the list of standard NSFW concepts
        for concet_idx in range(len(cos_dist[0])):
            concept_cos = cos_dist[i][concet_idx]
            concept_threshold = self.concept_embeds_weights[concet_idx].item()
            
            # Compute how much the current image matches the NSFW concept
            result_img["concept_scores"][concet_idx] = round(
                concept_cos - concept_threshold + adjustment, 3
            )
            
            # If there's a strong match, flag the image
            if result_img["concept_scores"][concet_idx] > 0:
                result_img["bad_concepts"].append(concet_idx)
                matches["nsfw"].append(concepts[concet_idx])

    # Check if any images have been flagged as NSFW
    has_nsfw_concepts = len(matches["nsfw"]) > 0

    # Return the matches and the flag status
    return matches, has_nsfw_concepts