import tempfile
import pytest
from cctools import CCTools
from pathlib import Path
import os,json

# 假设测试数据位置
TEST_ROOT = Path("tests/data/123")
TEST_ANNFILE = Path("instances_default.json")
TEST_IMGDIR = Path("images")
TEST_ANNDIR = Path("annotations")

@pytest.fixture
def ccdata_file():
    return CCTools(ROOT=TEST_ROOT, ANNFILE=TEST_ANNFILE, IMGDIR=TEST_IMGDIR, ANNDIR=TEST_ANNDIR)

@pytest.fixture
def ccdata_data():
    # 读取json文件
    with open(TEST_ROOT.joinpath(TEST_ANNDIR).joinpath(TEST_ANNFILE), "r") as f:
        ccdata = json.load(f)
    return CCTools(CCDATA=ccdata)

def test_cctools_init_with_file():
    cctools = CCTools(ROOT=TEST_ROOT, ANNFILE=TEST_ANNFILE, IMGDIR=TEST_IMGDIR, ANNDIR=TEST_ANNDIR)
    assert cctools.ANNFILE == TEST_ANNFILE
    assert cctools.IMGDIR == TEST_IMGDIR
    assert cctools.ANNDIR == TEST_ANNDIR
    
def test_cctools_init_with_ccdata():
    # 读取json文件
    with open(TEST_ROOT.joinpath(TEST_ANNDIR).joinpath(TEST_ANNFILE), "r") as f:
        ccdata = json.load(f)
    cctools = CCTools(CCDATA=ccdata)
    assert cctools.ANNFILE == TEST_ANNFILE
    assert cctools.IMGDIR == TEST_IMGDIR
    assert cctools.ANNDIR == TEST_ANNDIR

def test_cctools_init_with_ccdata_new_root():
    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        # 读取json文件
        with open(TEST_ROOT.joinpath(TEST_ANNDIR).joinpath(TEST_ANNFILE), "r") as f:
            ccdata = json.load(f)
        cctools = CCTools(CCDATA=ccdata,ROOT=temp_dir)
        
        
        # 检查临时文件是否存在
        assert Path(temp_dir).exists()
        assert cctools.ROOT == Path(temp_dir)
        assert cctools.ANNFILE == TEST_ANNFILE
        assert cctools.IMGDIR == TEST_IMGDIR
        assert cctools.ANNDIR == TEST_ANNDIR
        
    # 临时目录会在with语句结束后自动删除
    assert not Path(temp_dir).exists()
    
def test_visual(ccdata_file):
    ccdata_file.visual(overwrite=True)

def test_save(ccdata_file):
    # 创建临时目录
    with tempfile.TemporaryDirectory() as NEW_ROOT:
        NEW_ROOT = Path(NEW_ROOT)
        ccdata_file.save(New=CCTools(ROOT=NEW_ROOT),visual=True,overwrite=True)
        assert NEW_ROOT.exists()
        assert NEW_ROOT.joinpath(TEST_ANNDIR).exists()
        assert NEW_ROOT.joinpath(TEST_IMGDIR).exists()
        assert NEW_ROOT.joinpath(TEST_ANNDIR).joinpath(TEST_ANNFILE).exists()
    
    assert not NEW_ROOT.exists()
    
    