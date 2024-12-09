import tensorflow_datasets as tfds
import tensorflow as tf

class MyDataset(tfds.core.GeneratorBasedBuilder):
    """Dataset personnalisé pour MyDataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Version initiale.",
    }

    def _info(self):
        """Retourne les métadonnées du dataset."""
        return tfds.core.DatasetInfo(
            builder=self,
            description="Dataset B pour le pierre papier ciseaux",
            features=tfds.features.FeaturesDict({
                "image": tfds.features.Image(shape=(300, 300, 3)),
                "label": tfds.features.ClassLabel(names=["rock", "paper", "scissors"]),
            }),
            supervised_keys=("image", "label"),  # Clés pour le modèle supervisé
        )

    def _split_generators(self, dl_manager):
        """Définit les splits (train/test) et télécharge les données si nécessaire."""
        data_path = "./datasetB"
        return {
            "train": self._generate_examples(path=f"{data_path}/train"),
            "test": self._generate_examples(path=f"{data_path}/test"),
            "val": self._generate_examples(path=f"{data_path}/val"),
        }

    def _generate_examples(self, path):
        """Génère des exemples à partir des fichiers."""
        for idx, file_path in enumerate(tf.io.gfile.glob(f"{path}/*/*.jpg")):
            class_name = ""
            if ("rock" in file_path):
                class_name = "rock"
            elif ("paper" in file_path):
                class_name = "paper"
            else:
                class_name = "scissors"
            yield idx, {
                "image": file_path,
                "label": class_name,
            }