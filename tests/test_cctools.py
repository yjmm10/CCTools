
import os,json, tempfile, pytest
from cctools import CCTools
from pathlib import Path
import copy




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

@pytest.fixture
def merge_one1():
    return CCTools(ROOT=TEST_ROOT, ANNFILE=Path("instances_one1.json"), IMGDIR=TEST_IMGDIR, ANNDIR=TEST_ANNDIR)

@pytest.fixture
def merge_one2():
    return CCTools(ROOT=TEST_ROOT, ANNFILE=Path("instances_one2.json"), IMGDIR=TEST_IMGDIR, ANNDIR=TEST_ANNDIR)

@pytest.fixture
def split_file():
    return CCTools(ROOT=TEST_ROOT, ANNFILE=Path("instances_split.json"), IMGDIR=TEST_IMGDIR, ANNDIR=TEST_ANNDIR)


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
    

def test_static(ccdata_file):
    result = ccdata_file.static()
    assert result['total_images'] == len(ccdata_file.CCDATA.dataset['images'])
    assert result['total_annotations'] == len(ccdata_file.CCDATA.dataset['annotations'])
    
    
def test_update_cat(ccdata_file):
    newCat = {3:'Text',2:'Table',1:'Formula',4:'Figure'}
    ccdata_file.update_cat(newCat=newCat)
    assert {cat['id']: cat['name'] for cat in ccdata_file.CCDATA.dataset['categories']} == newCat 
    
def test_update_cat_with_id(ccdata_file):
    newCat = {3:'Text',2:'Table',1:'Formula',4:'Figure',5:'Image'}
    ccdata_file.update_cat(newCat=newCat)
    assert {cat['id']: cat['name'] for cat in ccdata_file.CCDATA.dataset['categories']} == newCat 

def test_rename_cat(ccdata_file):
    ccdata_file.rename_cat(old_name='Formula',new_name='Formula2')
    assert {cat['id']: cat['name'] for cat in ccdata_file.CCDATA.dataset['categories']} == {1: 'Text', 2: 'Table', 3: 'Formula2', 4: 'Figure'}

def test_get_cat(ccdata_file):
    assert ccdata_file._get_cat(cat='Figure') == (True,4)
    assert ccdata_file._get_cat(cat=4) == (True,'Figure')
    assert ccdata_file._get_cat(cat='Figure2') == (False,None)
    assert ccdata_file._get_cat(cat=5) == (False,None)
   

def test_get_img(ccdata_file):
    assert ccdata_file._get_img(img='a new precession formula(fukushima 2003)_5.jpg') == (True,1)    
    assert ccdata_file._get_img(img=1) == (True,'a new precession formula(fukushima 2003)_5.jpg')
    assert ccdata_file._get_img(img='a new precession formula(fukushima 2003)_5') == (True,1)    
    assert ccdata_file._get_img(img='a new precession formula(fukus') == (True,1)    

def test_updateIndex(merge_one2):
    merge_one2._updateIndex(imgIndex=5,annIndex=6)
    assert merge_one2.CCDATA.imgs[5]['id'] == 5
    assert merge_one2.CCDATA.anns[6]['id'] == 6
    
def test_get_imglist(merge_one2):
    result = merge_one2._get_imglist()
    assert len(result) == 1
    
def test_merge(merge_one1,merge_one2):
    merge_one1.merge(other=merge_one2,cat_keep=True)
    pass

def test_merge_without_overwrite(merge_one1,merge_one2):
    merge_one1.merge(other=merge_one2,cat_keep=True,overwrite=False)
    pass

def test_filter_and(ccdata_file):
    result = ccdata_file._filter(catIds=[1],imgIds=[1],annIds=[2])
    assert len(result) == 1

def test_filter_or(ccdata_file):
    # 获取所有包含catIds=1的结果加上所有所有imgIds=1的annIds
    result = ccdata_file._filter(catIds=[1],imgIds=[1],mod="or")
    assert len(result) == 12

def test_filter(ccdata_file):
    result = ccdata_file._filter(catIds=[1],imgIds=[1,2],annIds=[1,2])
    assert len(result) == 2

def test_filter_alignCat(ccdata_file):
    with tempfile.TemporaryDirectory() as temp_dir:
        newObj = CCTools(ROOT=Path(temp_dir))
        ccdata_file.filter(imgs=[1,],newObj=newObj,alignCat=True)
        assert newObj.ROOT == Path(temp_dir)
        assert len(newObj.CCDATA.dataset['categories']) == len(ccdata_file.CCDATA.dataset['categories'])
        
def test_filter_no_alignCat(ccdata_file):
    with tempfile.TemporaryDirectory() as temp_dir:
        newObj = CCTools(ROOT=Path(temp_dir))
        ccdata_file.filter(imgs=[1,],newObj=newObj,alignCat=False)
        assert newObj.ROOT == Path(temp_dir)
        assert len(newObj.CCDATA.dataset['images']) == 1

def test_get_imgIds_by_annIds(ccdata_file):
    result = ccdata_file._get_imgIds_by_annIds(annIds=[13,14])
    assert result == [2]

def test_filter_multi_cond(ccdata_file):

    with tempfile.TemporaryDirectory() as temp_dir:
        newObj = CCTools(ROOT=Path(temp_dir))
        ccdata_file.filter(imgs=["xx"],cats=["Formula"],newObj=newObj,mod="and",visual=True)
        assert newObj.ROOT == Path(temp_dir)
        assert len(newObj.CCDATA.dataset['images']) == 1


def test_filter_multi_cond_or(ccdata_file):
    with tempfile.TemporaryDirectory() as temp_dir:
        newObj = CCTools(ROOT=Path(temp_dir))
        ccdata_file.filter(imgs=["xx"],cats=["Formula"],newObj=newObj,mod="or",visual=True)
        assert newObj.ROOT == Path(temp_dir)
        assert len(newObj.CCDATA.dataset['images']) == 2

def test_filter_multi_cond_level_ann(ccdata_file):
    with tempfile.TemporaryDirectory() as temp_dir:
        newObj = CCTools(ROOT=Path(temp_dir))
        ccdata_file.filter(imgs=["xx"],cats=["Formula"],newObj=newObj,mod="and",visual=True,level="ann")
        assert newObj.ROOT == Path(temp_dir)
        assert len(newObj.CCDATA.dataset['images']) == 1
        assert len(newObj.CCDATA.dataset['annotations']) == 1
        
def test_filter_sep_level_img(ccdata_file):
    with tempfile.TemporaryDirectory() as temp_dir:
        newObj = CCTools(ROOT=Path(temp_dir))
        ccdata_file.filter(imgs=["xx"],cats=["Formula"],newObj=newObj,visual=True,level="img",sep_data=True)
        assert newObj.ROOT == Path(temp_dir)
        assert len(newObj.CCDATA.dataset['images']) == 1
        assert len(ccdata_file.CCDATA.dataset['images']) == 1
        
        with tempfile.TemporaryDirectory() as temp_dir2:
            newObj.save(New=CCTools(ROOT=Path(temp_dir2)),visual=True)
            assert Path(temp_dir2).exists()
        
def test_filter_sep_leve_ann(ccdata_file):
    with tempfile.TemporaryDirectory() as temp_dir:
        newObj = CCTools(ROOT=Path(temp_dir))
        ccdata_file.filter(imgs=["xx"],cats=["Formula"],newObj=newObj,visual=True,level="ann",sep_data=True)
        assert newObj.ROOT == Path(temp_dir)
        assert len(newObj.CCDATA.dataset['images']) == 1
    
    with tempfile.TemporaryDirectory() as temp_dir2:
        newObj.save(New=CCTools(ROOT=Path(temp_dir2)),visual=True)
        assert Path(temp_dir2).exists()
        
def test_filter_sep_level_img_alignCat_or(ccdata_file):
    with tempfile.TemporaryDirectory() as temp_dir:
        newObj = CCTools(ROOT=Path(temp_dir))
        ccdata_file.filter(imgs=["xx"],cats=["Formula"],mod="or",newObj=newObj,visual=True,level="img",sep_data=True,alignCat=True)
        assert newObj.ROOT == Path(temp_dir)
        assert len(newObj.CCDATA.dataset['images']) == 2
        
        with tempfile.TemporaryDirectory() as temp_dir2:
            newObj.save(New=CCTools(ROOT=Path(temp_dir2)),visual=True)
            assert Path(temp_dir2).exists()
            assert len(newObj.CCDATA.dataset['images']) == 2
            
            
def test_correct(ccdata_file):
    with tempfile.TemporaryDirectory() as temp_dir:
        newObj = CCTools(ROOT=Path(temp_dir))
        ccdata_file.correct(api_url=lambda x,y:1,cats=["Formula"],newObj=newObj)
        assert newObj.ROOT == Path(temp_dir)
        assert len(newObj.CCDATA.dataset['annotations']) == 1
    pass

def test_split(split_file):
    split_file.split(ratio=[0.6,0.4],by_file=True)