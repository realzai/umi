from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
import matplotlib.pyplot as plt

if __name__ == "__main__":
    model = UNet2DModel.from_pretrained("zaibutcooler/umi")
    scheduler = DDPMScheduler.from_pretrained("zaibutcooler/umi")

    pipeline = DDPMPipeline(
        unet=model,
        scheduler=scheduler,
        safety_checker=None,
        requires_safety_checker=False,
    )

    images = pipeline.run(batch_size=4).images

    for i, img in enumerate(images):
        plt.subplot(1, 4, i + 1)
        plt.imshow(img)
        plt.axis("off")
    plt.show()
