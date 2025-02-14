import torch
import torch.nn.functional as F

class QuaternionOps:
    @staticmethod
    def compute_rotation_quaternion(v1, v2):
        """
        Compute rotation quaternion between vector sets
        v1, v2: [B, C, T, N] tensors where C=3 for xyz coordinates
        Returns: [B, 4, T, N] quaternion
        """
        device = v1.device
        B, C, T, N = v1.shape
        
        # Reshape to [B*T*N, 3]
        v1_flat = v1.permute(0, 2, 3, 1).reshape(-1, 3)
        v2_flat = v2.permute(0, 2, 3, 1).reshape(-1, 3)
        
        # Normalize vectors
        v1_norm = F.normalize(v1_flat, dim=1)
        v2_norm = F.normalize(v2_flat, dim=1)
        
        # Compute rotation axis
        axis = torch.cross(v1_norm, v2_norm, dim=1)  # [B*T*N, 3]
        axis_norm = torch.norm(axis, dim=1, keepdim=True)
        
        # Handle parallel vectors
        parallel_mask = (axis_norm < 1e-6).squeeze(1)
        default_axis = torch.tensor([[1., 0., 0.]], device=device).expand(v1_flat.shape[0], -1)
        axis = torch.where(parallel_mask.unsqueeze(1), default_axis, F.normalize(axis, dim=1))
        
        # Compute rotation angle
        dot = torch.sum(v1_norm * v2_norm, dim=1, keepdim=True)
        dot = torch.clamp(dot, -1 + 1e-6, 1 - 1e-6)
        angle = torch.acos(dot)
        
        # Create quaternion [w, x, y, z]
        half_angle = angle * 0.5
        cos_half = torch.cos(half_angle)
        sin_half = torch.sin(half_angle)
        quat = torch.cat([cos_half, axis * sin_half], dim=1)  # [B*T*N, 4]
        
        # Reshape back to [B, 4, T, N]
        quat = quat.reshape(B, T, N, 4).permute(0, 3, 1, 2)
        
        return quat
    
    @staticmethod
    def quaternion_multiply(q1, q2):
        """
        Multiply two quaternions
        Input shapes: [B, 4, T, N] or any shape where dim=1 is quaternion components
        """
        w1, x1, y1, z1 = q1.unbind(1)
        w2, x2, y2, z2 = q2.unbind(1)
        
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        
        return torch.stack([w, x, y, z], dim=1)
    
    def compute_motion_consistency(self, pos_prev, pos_curr):
        """Compute cycle consistency error and forward rotation"""
        # Ensure inputs are on the same device
        device = pos_prev.device
        assert pos_curr.device == device, "Input tensors must be on the same device"
        
        # Compute quaternions
        q_forward = self.compute_rotation_quaternion(pos_prev, pos_curr)
        q_backward = self.compute_rotation_quaternion(pos_curr, pos_prev)
        
        # Create identity quaternion [1, 0, 0, 0]
        identity = torch.zeros_like(q_forward)
        identity[:, 0] = 1  # First component is w
        
        # Compute cycle consistency
        q_cycle = self.quaternion_multiply(q_forward, q_backward)
        consistency_error = torch.norm(q_cycle - identity, dim=1)
        
        return consistency_error, q_forward  
    
    
def test_4d_input():
    # Test parameters
    batch_size = 2
    timesteps = 8
    num_points = 16
    
    # Create test inputs
    v1 = torch.randn(batch_size, 3, timesteps, num_points).cuda()
    v2 = torch.randn(batch_size, 3, timesteps, num_points).cuda()
    
    # Compute quaternions
    quat_ops = QuaternionOps()
    rotation_quat = quat_ops.compute_rotation_quaternion(v1, v2)
    
    print(f"Input shape: {v1.shape}")
    print(f"Output quaternion shape: {rotation_quat.shape}")
    print(f"Sample quaternion values:\n{rotation_quat[0, :, 0, 0]}")  # First quaternion of first batch
    
    # Check output shape
    assert rotation_quat.shape == (batch_size, 4, timesteps, num_points)
    print("Shape test passed!")
    
    # Check quaternion normalization
    quat_norms = torch.norm(rotation_quat, dim=1)
    assert torch.allclose(quat_norms, torch.ones_like(quat_norms), atol=1e-6)
    print("Normalization test passed!")

if __name__ == "__main__":
    test_4d_input()