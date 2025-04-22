from pathlib import Path
import urllib.request
import zipfile
import tarfile

import pytest

@pytest.fixture(scope="session")
def test_data_dir(pytestconfig):
    return pytestconfig.cache.mkdir("openequivariance")

@pytest.fixture(scope="session")
def nequip_test_data_dir(test_data_dir):
    path = test_data_dir / "nequip"
    path.mkdir(exist_ok=True)
    return path

@pytest.fixture(scope="session")
def nequip_ccsd_test_data_dir(nequip_test_data_dir):
    path = nequip_test_data_dir / "ccsd"
    path.mkdir(exist_ok=True)

    for sub_dir_name, url in [
        ("aspirin_ccsd",         "http://www.quantum-machine.org/gdml/data/xyz/aspirin_ccsd.zip"), 
        ("benzene_ccsd_t",       "http://www.quantum-machine.org/gdml/data/xyz/benzene_ccsd_t.zip"),
        ("malonaldehyde_ccsd_t", "http://www.quantum-machine.org/gdml/data/xyz/malonaldehyde_ccsd_t.zip"),
        ("toluene_ccsd_t",       "http://www.quantum-machine.org/gdml/data/xyz/toluene_ccsd_t.zip"),
        ("ethanol_ccsd_t",       "http://www.quantum-machine.org/gdml/data/xyz/ethanol_ccsd_t.zip"),
    ]:
        download_and_unzip_if_necessary(path / sub_dir_name, url)

    return path

@pytest.fixture(scope="session")
def nequip_rmd17_test_data_dir(nequip_test_data_dir):
    path = nequip_test_data_dir / "rmd17"
    download_and_unzip_if_necessary(path, "https://figshare.com/ndownloader/articles/12672038/versions/3")
    tarfile_path = (path / "rmd17.tar.bz2")
    if tarfile_path.exists():
        with tarfile.open(tarfile_path, "r:bz2") as tar: 
            tar.extractall(path=nequip_test_data_dir, filter='data')
        
        tarfile_path.unlink()
    return path


def download_and_unzip_if_necessary(unzip_path : Path, url):
    if unzip_path.is_dir():
        return unzip_path
    else:
        unzip_path.mkdir()
        zip_path = unzip_path.parent / "temp.zip"
        
        with urllib.request.urlopen(url) as response, zip_path.open('wb') as out_file:
            while chunk := response.read(8192):
                out_file.write(chunk)         

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
        
        zip_path.unlink()
        return unzip_path
    
    
def test_nequip_ccsd_test_data_dir(nequip_ccsd_test_data_dir):
    assert nequip_ccsd_test_data_dir.is_dir()

def test_nequip_rmd17_test_data_dir(nequip_rmd17_test_data_dir):
    assert nequip_rmd17_test_data_dir.is_dir()       
