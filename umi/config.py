class Config:
    def __init__(self,img_size=128,gray_scale=False,channels=3) -> None:
        self.img_size:int = img_size
        self.gray_scale:bool = gray_scale
        self.channels:int = channels