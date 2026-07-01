import torch
import torch.nn as nn


class PlateLoss(nn.Module):
    """
    Multi-task Loss

    Classification :
        CrossEntropyLoss

    Polygon :
        MSELoss
    """

    def __init__(
        self,
        cls_weight=1.0,
        polygon_weight=5.0,
    ):

        super().__init__()

        self.cls_weight = cls_weight
        self.polygon_weight = polygon_weight

        self.cls_loss_fn = nn.CrossEntropyLoss()

        self.polygon_loss_fn = nn.MSELoss()

    # ---------------------------------------------------------

    def forward(self, output, batch):

        class_logits = output["class_logits"]

        polygon_pred = output["polygon"]

        class_target = batch["class"]

        polygon_target = batch["polygon"]

        # --------------------------------------------

        cls_loss = self.cls_loss_fn(
            class_logits,
            class_target,
        )

        polygon_loss = self.polygon_loss_fn(
            polygon_pred,
            polygon_target,
        )

        total_loss = (
            self.cls_weight * cls_loss
            +
            self.polygon_weight * polygon_loss
        )

        return {

            "classification_loss": cls_loss,

            "polygon_loss": polygon_loss,

            "total_loss": total_loss,

        }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    batch_size = 8

    output = {

        "class_logits": torch.randn(batch_size, 2),

        "polygon": torch.rand(batch_size, 8),

    }

    batch = {

        "class": torch.randint(
            0,
            2,
            (batch_size,),
        ),

        "polygon": torch.rand(
            batch_size,
            8,
        ),
    }

    criterion = PlateLoss()

    losses = criterion(
        output,
        batch,
    )

    print("=" * 60)

    print("LOSS TEST")

    print("=" * 60)

    print()

    for name, value in losses.items():

        print(f"{name:25s}: {value.item():.6f}")