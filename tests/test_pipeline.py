
import os,json, tempfile, pytest
from cctools import Pipeline,CCTools
from pathlib import Path



# 假设测试数据位置
ROOT = Path("tests/data/pipeline")
ANNFILE = Path("instances_default.json")
ANNDIR = Path("annotations")
IMGDIR = Path("images")

CACHE_PATH = Path(".cache")

TRAIN_IMGDIR = Path("Train")
VAL_IMGDIR = Path("Val")
TEST_IMGDIR = Path("Test")

@pytest.fixture
def ccdata_file():
    return CCTools(ROOT=ROOT, ANNFILE=ANNFILE, IMGDIR=IMGDIR, ANNDIR=ANNDIR)

def test_split_data(ccdata_file):
    # by_file=False
    with tempfile.TemporaryDirectory() as NEW_ROOT:
        empty_data = CCTools(ROOT=NEW_ROOT, ANNFILE=ANNFILE, IMGDIR=IMGDIR, ANNDIR=ANNDIR,empty_init=True)
        pipeline = Pipeline()
        config = {  'cache_path':CACHE_PATH,
                    'cctools':ccdata_file,
                    'ratio':[0.7,0.2,0.1],
                    'by_file':False,
                    'visual':False,
                    'newObj':empty_data
                    }
    
        pipeline.split_data(config=config)
        trainObj = CCTools(ROOT=empty_data.ROOT,IMGDIR="Train",ANNFILE="instances_Train.json")
        valObj = CCTools(ROOT=empty_data.ROOT,IMGDIR="Val",ANNFILE="instances_Val.json")  
        testObj = CCTools(ROOT=empty_data.ROOT,IMGDIR="Test",ANNFILE="instances_Test.json")
        
        assert len(trainObj.CCDATA.dataset["images"]) == int(config['ratio'][0]*10)
        assert len(valObj.CCDATA.dataset["images"]) == int(config['ratio'][1]*10)
        assert len(testObj.CCDATA.dataset["images"]) == int(config['ratio'][2]*10)
    
    # by_file=True
    with tempfile.TemporaryDirectory() as NEW_ROOT:
        empty_data = CCTools(ROOT=NEW_ROOT, ANNFILE=ANNFILE, IMGDIR=IMGDIR, ANNDIR=ANNDIR,empty_init=True)
        pipeline = Pipeline()
        config = {  'cache_path':CACHE_PATH,
                    'cctools':ccdata_file,
                    'ratio':[0.7,0.2,0.1],
                    'by_file':True,
                    'visual':False,
                    'newObj':empty_data
                    }
    
        pipeline.split_data(config=config)
        trainObj = CCTools(ROOT=empty_data.ROOT,IMGDIR="Train",ANNFILE="instances_Train.json")
        valObj = CCTools(ROOT=empty_data.ROOT,IMGDIR="Val",ANNFILE="instances_Val.json")  
        testObj = CCTools(ROOT=empty_data.ROOT,IMGDIR="Test",ANNFILE="instances_Test.json")
        
        assert len(trainObj.CCDATA.dataset["images"]) == 6
        assert len(valObj.CCDATA.dataset["images"]) == 2
        assert len(testObj.CCDATA.dataset["images"]) == 2
    

# 用来制造测试数据
# def test_split_data1(ccdata_file):
    
#     cctools = CCTools(ROOT="split_test", ANNFILE="instances_Val.json", IMGDIR=IMGDIR, ANNDIR=ANNDIR)
    
#     newObj = CCTools(ROOT="split_test_new", ANNFILE="instances_Val.json", IMGDIR=IMGDIR, ANNDIR=ANNDIR,CP_IMGPATH=cctools.ROOT.joinpath(IMGDIR),empty_init=True)
#     cctools.filter(imgs=[1,2,3,4,5,6,7,8,9,10,11,12,15,17,24,28,30],newObj=newObj,visual=False,mod="and",alignCat=True,sep_data=False,level="img")
    

def test_static():
    pipeline = Pipeline()
    result,stats = pipeline.static_data(ROOT.parent,target_dirs=["annotations","images"],data_match={"instances_default_1.json":"images","instances_default.json":"images","instances_default.json":"images"})
    assert result is not None
    assert stats is not None

    

    