import torch

if __name__ == '__main__':
    checkpoint = 'BEST_checkpoint.tar'
    checkpoint = torch.load(checkpoint)
    model = checkpoint['model']

    torch.save(model.state_dict(), 'speaker-embeddings.pt')
