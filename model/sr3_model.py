# ... 现有代码 ...

def set_zsample_params(self, use_zsample=False, gamma1=5.5, gamma2=0.0):
    """设置Z-Sampling参数
    
    Args:
        use_zsample (bool): 是否使用Z-Sampling
        gamma1 (float): 去噪过程的引导尺度
        gamma2 (float): 反转过程的引导尺度
    """
    self.use_zsample = use_zsample
    self.gamma1 = gamma1
    self.gamma2 = gamma2
    self.logger.info(f'Z-Sampling参数设置: use_zsample={use_zsample}, gamma1={gamma1}, gamma2={gamma2}')

# ... 现有代码 ...

def test(self, continous=False):
    self.netG.eval()
    with torch.no_grad():
        if continous:
            if not self.use_zsample:
                # 原始采样方法
                self.output = self.netG(self.lq, continous=True)
            else:
                # Z-Sampling方法
                self.output = self.zsample_inference(self.lq)
        else:
            self.output = self.netG(self.lq)
            
    self.netG.train()

def zsample_inference(self, lq):
    """使用Z-Sampling策略进行推理
    
    Args:
        lq (Tensor): 低质量输入图像
        
    Returns:
        Tensor: 生成的高质量图像
    """
    # 获取模型参数
    model = self.netG
    t_size = model.module.timesteps if hasattr(model, 'module') else model.timesteps
    
    # 初始化噪声
    x_t = torch.randn_like(lq)
    
    # 存储中间结果
    intermediates = []
    
    # Z-Sampling过程
    for t in range(t_size, 0, -1):
        t_tensor = torch.full((lq.shape[0],), t-1, device=lq.device, dtype=torch.long)
        
        # 1. 去噪步骤 (使用gamma1)
        with torch.no_grad():
            # 设置强引导
            model.set_guidance_scale(self.gamma1)
            x_t_minus_1 = model.denoise_step(x_t, t_tensor, lq)
            intermediates.append(x_t_minus_1)
        
        # 2. 反转步骤 (使用gamma2)
        with torch.no_grad():
            # 设置弱引导
            model.set_guidance_scale(self.gamma2)
            x_t_tilde = model.reverse_step(x_t_minus_1, t_tensor, lq)
        
        # 3. 使用反转后的结果替换原始噪声
        x_t = x_t_tilde
    
    # 返回所有中间结果和最终结果
    return torch.stack(intermediates)

# ... 现有代码 ...