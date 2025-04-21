# 在UNet模型中添加以下方法

def p_sample_with_encoder_features(self, x, t, condition_x=None, return_encoder_features=False):
    """执行一步采样并返回编码器特征
    
    Args:
        x: 当前噪声图像
        t: 当前时间步
        condition_x: 条件输入
        return_encoder_features: 是否返回编码器特征
        
    Returns:
        tuple: (去噪后的图像, 编码器特征字典)
    """
    # 存储编码器特征
    encoder_features = {}
    
    # 执行UNet的编码器部分，并保存特征
    h = x
    for i, module in enumerate(self.input_blocks):
        h = module(h, t)
        # 保存编码器特征
        encoder_features[f"encoder_{i}"] = h
    
    # 执行中间块
    h = self.middle_block(h, t)
    encoder_features["middle"] = h
    
    # 执行解码器部分
    for i, module in enumerate(self.output_blocks):
        # 获取跳跃连接
        skip_connection = encoder_features[f"encoder_{len(self.input_blocks) - 1 - i}"]
        h = torch.cat([h, skip_connection], dim=1)
        h = module(h, t)
    
    # 最终输出
    h = self.out(h)
    
    # 计算去噪后的图像
    out = self.predict_start_from_noise(x, t, h)
    
    # 应用DDIM采样公式
    x_prev = self.q_posterior(x, out, t)
    
    if return_encoder_features:
        return x_prev, encoder_features
    else:
        return x_prev

def p_sample_with_cached_encoder_features(self, x, t, encoder_features, condition_x=None):
    """使用缓存的编码器特征执行一步采样
    
    Args:
        x: 当前噪声图像
        t: 当前时间步
        encoder_features: 缓存的编码器特征
        condition_x: 条件输入
        
    Returns:
        Tensor: 去噪后的图像
    """
    # 直接使用中间特征
    h = encoder_features["middle"]
    
    # 执行解码器部分
    for i, module in enumerate(self.output_blocks):
        # 获取缓存的跳跃连接
        skip_connection = encoder_features[f"encoder_{len(self.input_blocks) - 1 - i}"]
        h = torch.cat([h, skip_connection], dim=1)
        h = module(h, t)
    
    # 最终输出
    h = self.out(h)
    
    # 计算去噪后的图像
    out = self.predict_start_from_noise(x, t, h)
    
    # 应用DDIM采样公式
    x_prev = self.q_posterior(x, out, t)
    
    return x_prev