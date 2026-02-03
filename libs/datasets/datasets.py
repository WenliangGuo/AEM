import torch
from .data_utils import trivial_batch_collator
datasets = {}
def register_dataset(name):
   def decorator(cls):
       datasets[name] = cls
       return cls
   return decorator

def make_dataset(name, is_training, clip_model, split, use_gcn, **kwargs):
   """
       A simple dataset builder
   """
   dataset = datasets[name](is_training, split, clip_model, use_gcn, **kwargs)
   return dataset

def make_data_loader(dataset, is_training, generator, batch_size, num_workers):
    """
        A simple dataloder builder
    """
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=trivial_batch_collator,
        shuffle=is_training,
        drop_last=is_training,
        generator=generator,
        persistent_workers=True
    )
    return loader
