import torch
import torch.nn.functional as F

def distillation_loss(
	student_logits: torch.Tensor,
	teacher_logits: torch.Tensor,
	temperature: float = 1.0
) -> torch.Tensor:
	"""
	Compute knowledge distillation loss using PyTorch.
	
	Args:
		student_logits: Logits from student model
		teacher_logits: Logits from teacher model
		temperature: Softmax temperature
		
	Returns:
		Distillation loss
	"""
	# Your code here
    T = temperature
	return T ** 2 * F.kl_div(
        F.log_softmax(student_logits / T, dim=-1), 
        F.log_softmax(teacher_logits / T, dim=-1), 
        reduction="sum", log_target=True)