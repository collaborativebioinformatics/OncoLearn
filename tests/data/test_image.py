import pytest
import torch
from pathlib import Path
from PIL import Image

from oncolearn.data.modalities.image.loaders.dicom_loader import DicomLoader
from oncolearn.data.modalities.image.loaders.pillow_loader import PillowLoader
from oncolearn.data.modalities.image.dataset import ImageDataset, ImageDataModule


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_png(path: Path, size=(10, 10), color=(100, 150, 200)):
    """Write a minimal solid-color RGB PNG."""
    Image.new("RGB", size, color=color).save(path)


@pytest.fixture()
def brca_image_dir(tmp_path):
    """
    Populate a TCIA-like directory structure with synthetic PNG slices.

    Layout: <data_dir>/TCGA-BRCA/<patient_id>/<slice>.png

    6 patients, each with a different number of slices so that both the
    "more slices than n_slices" and "fewer slices than n_slices" paths
    are exercised.
    """
    patients = {
        "TCGA-A1-0001": 8,  # more than n_slices=5
        "TCGA-A1-0002": 3,  # fewer than n_slices=5
        "TCGA-A2-0001": 5,  # exactly n_slices=5
        "TCGA-A2-0002": 6,
        "TCGA-B1-0001": 7,
        "TCGA-B1-0002": 4,
    }

    cohort_dir = tmp_path / "tcia" / "TCGA-BRCA"
    for patient_id, n in patients.items():
        patient_dir = cohort_dir / patient_id
        patient_dir.mkdir(parents=True)
        for i in range(n):
            _make_png(patient_dir / f"slice_{i:03d}.png", color=(i * 10, 50, 100))

    return tmp_path / "tcia"


@pytest.fixture()
def sample_png(tmp_path) -> Path:
    """A single valid PNG file."""
    p = tmp_path / "sample.png"
    _make_png(p, size=(32, 32))
    return p


# ---------------------------------------------------------------------------
# Loader: PillowLoader
# ---------------------------------------------------------------------------

def test_pillow_loader_can_load():
    assert PillowLoader.can_load(Path("img.png")) is True
    assert PillowLoader.can_load(Path("img.jpg")) is True
    assert PillowLoader.can_load(Path("img.jpeg")) is True
    assert PillowLoader.can_load(Path("img.tiff")) is True
    assert PillowLoader.can_load(Path("img.bmp")) is True
    assert PillowLoader.can_load(Path("img.dcm")) is False
    assert PillowLoader.can_load(Path("img.csv")) is False


def test_pillow_loader_returns_rgb_image(sample_png):
    img = PillowLoader.load(sample_png)
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"


def test_pillow_loader_missing_pillow(monkeypatch):
    """Raises ImportError with a helpful message when PIL is unavailable."""
    import PIL.Image

    def _raise(*_args, **_kwargs):
        raise ImportError("No module named 'PIL'")

    monkeypatch.setattr(PIL.Image, "open", _raise)

    with pytest.raises(ImportError, match="Pillow is required for standard image files"):
        PillowLoader.load(Path("fake.png"))


# ---------------------------------------------------------------------------
# Loader: DicomLoader
# ---------------------------------------------------------------------------

def test_dicom_loader_can_load():
    assert DicomLoader.can_load(Path("scan.dcm")) is True
    assert DicomLoader.can_load(Path("scan.dicom")) is True
    assert DicomLoader.can_load(Path("scan.DICOM")) is True
    assert DicomLoader.can_load(Path("scan.png")) is False


def test_dicom_loader_missing_pydicom(monkeypatch):
    """Raises ImportError with a helpful message when pydicom is unavailable."""
    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "pydicom":
            raise ImportError("No module named 'pydicom'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    with pytest.raises(ImportError, match="pydicom and SimpleITK required"):
        DicomLoader.load(Path("fake.dcm"))


# ---------------------------------------------------------------------------
# ImageDataset
# ---------------------------------------------------------------------------

def _make_dataset(tmp_path, n_patients=4, slices_per_patient=6, n_slices=5):
    """Build an ImageDataset backed by real PNG files."""
    patient_ids = [f"TCGA-XX-{i:04d}" for i in range(n_patients)]
    patient_to_files = {}
    for pid in patient_ids:
        patient_dir = tmp_path / pid
        patient_dir.mkdir()
        files = []
        for j in range(slices_per_patient):
            p = patient_dir / f"slice_{j:03d}.png"
            _make_png(p)
            files.append(p)
        patient_to_files[pid] = files

    return ImageDataset(patient_to_files, patient_ids, transform=None, n_slices=n_slices)


def test_image_dataset_len(tmp_path):
    ds = _make_dataset(tmp_path, n_patients=5)
    assert len(ds) == 5


def test_image_dataset_get_keys_matches_patient_ids(tmp_path):
    ds = _make_dataset(tmp_path, n_patients=3)
    assert ds.get_keys() == ds.patient_ids


def test_image_dataset_output_shape(tmp_path):
    """Items should be (n_slices, 3, H, W) float32 tensors."""
    n_slices = 5
    ds = _make_dataset(tmp_path, n_patients=2, slices_per_patient=8, n_slices=n_slices)
    item = ds[0]

    assert "image" in item
    assert isinstance(item["image"], torch.Tensor)
    assert item["image"].ndim == 4
    assert item["image"].shape[0] == n_slices
    assert item["image"].shape[1] == 3   # RGB channels
    assert item["image"].dtype == torch.float32


def test_image_dataset_fewer_files_than_slices(tmp_path):
    """When a patient has fewer files than n_slices, sampling clips to valid indices."""
    ds = _make_dataset(tmp_path, n_patients=1, slices_per_patient=2, n_slices=5)
    item = ds[0]
    assert item["image"].shape[0] == 5


def test_image_dataset_zero_tensor_for_missing_patient():
    """A patient with no files falls back to an all-zero tensor."""
    ds = ImageDataset(
        patient_to_files={"TCGA-XX-0001": []},
        patient_ids=["TCGA-XX-0001"],
        n_slices=5,
    )
    item = ds[0]
    assert item["image"].shape == (5, 3, 224, 224)
    assert item["image"].sum().item() == pytest.approx(0.0)


def test_image_dataset_patient_id_in_item(tmp_path):
    ds = _make_dataset(tmp_path, n_patients=2)
    item = ds[0]
    assert "patient_id" in item
    assert item["patient_id"] == ds.patient_ids[0]


# ---------------------------------------------------------------------------
# ImageDataModule — BRCA-style integration
# ---------------------------------------------------------------------------

def test_image_datamodule_setup_discovers_patients(brca_image_dir):
    dm = ImageDataModule(
        tcia_cohort_name="BRCA",
        data_dir=str(brca_image_dir),
        image_size=(32, 32),
        batch_size=2,
        num_workers=0,
    )
    dm.setup()

    assert len(dm.patient_ids) == 6


def test_image_datamodule_setup_maps_correct_file_counts(brca_image_dir):
    """Each patient_to_files entry should have the right number of slices."""
    dm = ImageDataModule(
        tcia_cohort_name="BRCA",
        data_dir=str(brca_image_dir),
        image_size=(32, 32),
        batch_size=2,
        num_workers=0,
    )
    dm.setup()

    expected = {
        "TCGA-A1-0001": 8,
        "TCGA-A1-0002": 3,
        "TCGA-A2-0001": 5,
        "TCGA-A2-0002": 6,
        "TCGA-B1-0001": 7,
        "TCGA-B1-0002": 4,
    }
    for patient_id, count in expected.items():
        assert len(dm.patient_to_files[patient_id]) == count, patient_id


def test_image_datamodule_train_val_split(brca_image_dir):
    """80/20 split over 6 patients → 4 train, 2 val."""
    dm = ImageDataModule(
        tcia_cohort_name="BRCA",
        data_dir=str(brca_image_dir),
        image_size=(32, 32),
        batch_size=2,
        num_workers=0,
        train_split=0.8,
        seed=42,
    )
    dm.setup()

    train_n = len(dm.train_dataset)
    val_n = len(dm.val_dataset)

    assert train_n + val_n == 6
    assert train_n == 4
    assert val_n == 2


def test_image_datamodule_no_overlap_between_splits(brca_image_dir):
    """Train and val patient sets must be disjoint."""
    dm = ImageDataModule(
        tcia_cohort_name="BRCA",
        data_dir=str(brca_image_dir),
        image_size=(32, 32),
        batch_size=2,
        num_workers=0,
        seed=0,
    )
    dm.setup()

    train_ids = set(dm.train_dataset.patient_ids)
    val_ids = set(dm.val_dataset.patient_ids)
    assert train_ids.isdisjoint(val_ids)


def test_image_datamodule_datasets_return_tensors(brca_image_dir):
    """Items from train_dataset should be float32 image tensors."""
    dm = ImageDataModule(
        tcia_cohort_name="BRCA",
        data_dir=str(brca_image_dir),
        image_size=(32, 32),
        batch_size=2,
        num_workers=0,
    )
    dm.setup()

    item = dm.train_dataset[0]
    assert isinstance(item["image"], torch.Tensor)
    assert item["image"].dtype == torch.float32
    assert item["image"].shape == (5, 3, 32, 32)


def test_image_datamodule_dataloader_batches(brca_image_dir):
    """train_dataloader should yield a batch shaped (B, N, 3, H, W)."""
    dm = ImageDataModule(
        tcia_cohort_name="BRCA",
        data_dir=str(brca_image_dir),
        image_size=(32, 32),
        batch_size=2,
        num_workers=0,
    )
    dm.setup()

    loader = dm.train_dataloader()
    batch = next(iter(loader))

    assert "image" in batch
    B, N, C, H, W = batch["image"].shape
    assert B == 2
    assert N == 5
    assert C == 3
    assert H == W == 32


# ---------------------------------------------------------------------------
# ImageDataModule._extract_patient_id
# ---------------------------------------------------------------------------

def test_extract_patient_id_from_tcga_path(tmp_path):
    dm = ImageDataModule(tcia_cohort_name="BRCA", data_dir=str(tmp_path), num_workers=0)
    path = Path("/data/tcia/TCGA-BRCA/TCGA-A1-0001/scan/slice_001.png")
    assert dm._extract_patient_id(path) == "TCGA-A1-0001"


def test_extract_patient_id_falls_back_to_stem(tmp_path):
    """Paths with no 3-part TCGA segment fall back to the file stem."""
    dm = ImageDataModule(tcia_cohort_name="BRCA", data_dir=str(tmp_path), num_workers=0)
    path = Path("/data/images/no_patient_info/slice.png")
    assert dm._extract_patient_id(path) == "slice"
