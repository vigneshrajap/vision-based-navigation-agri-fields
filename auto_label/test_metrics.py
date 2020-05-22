import pytest
from pytest import approx
import numpy as np
from metrics import *

@pytest.fixture
def gt_with_zero():
    tmp = np.zeros(30)
    tmp[10:20] = 1
    tmp[20:30] = 2
    return tmp
    
@pytest.fixture
def gt_imbalanced():
    tmp = np.zeros(40)
    tmp[10:30] = 1
    tmp[30:40] = 2
    return tmp

@pytest.fixture
def pr_with_zero_misaligned():
    tmp = np.zeros(30)
    tmp[5:15] = 1
    tmp[15:25] = 2
    return tmp

@pytest.fixture
def pr_with_zero_perfect():
    tmp = np.zeros(30)
    tmp[10:20] = 1
    tmp[20:30] = 2
    return tmp

def test_1_iou_perfect_ignore_zero(gt_with_zero, pr_with_zero_perfect):

    cl_wise, mean, freq_w = get_iou(gt_with_zero, pr_with_zero_perfect, 3, ignore_zero_class=True)
    assert(
        np.isnan(cl_wise[0]) 
        and np.all(np.equal(cl_wise[1:2],[approx(1),approx(1)])) 
        and mean == approx(1) 
        and freq_w == approx(1))

def test_2_iou_misaligned_ignore_zero(gt_with_zero, pr_with_zero_misaligned):
    cl_wise, mean, freq_w = get_iou(gt_with_zero, pr_with_zero_misaligned, 3, ignore_zero_class=True)
    print(cl_wise, mean, freq_w)
    assert(
        np.isnan(cl_wise[0]) 
        and np.all(np.equal(cl_wise[1:3],[approx(1/2),approx(1/3)])) 
        and mean == approx((1/2+1/3)/2) 
        and freq_w == approx((1/2+1/3)/2)
        )

def test_3_iou_pr_all1_ignore_zero(gt_with_zero):

    cl_wise, mean, freq_w = get_iou(gt_with_zero, np.ones(30), 3, ignore_zero_class=True)
    assert(
        np.isnan(cl_wise[0]) 
        and np.all(np.equal(cl_wise[1:3],[approx(1/2),approx(0)]))
        and mean == approx((1/2)/2) 
        and freq_w == approx((1/2)/2)
    )

def test_4_iou_gt_all1_ignore_zero(pr_with_zero_perfect):
    cl_wise, mean, freq_w = get_iou(np.ones(30), pr_with_zero_perfect, 3, ignore_zero_class=True)
    print(mean,freq_w)
    assert(
        np.isnan(cl_wise[0]) 
        and np.all(np.equal(cl_wise[1:3],[approx(1/3),approx(0)]))
        and mean == approx((1/3)/2) 
        and freq_w == approx((1/3)/1)
    )

def test_5_iou_pr_all1_imbalanced_ignore_zero(gt_imbalanced):
    cl_wise, mean, freq_w = get_iou(gt_imbalanced, np.ones(40), 3, ignore_zero_class=True)
    assert(
        np.isnan(cl_wise[0]) 
        and np.all(np.equal(cl_wise[1:3],[approx(2/3),approx(0)]))
        and mean == approx((2/3)/2) 
        and freq_w == approx((2/3)*(2/3))
    )

def test_6_iou_all_wrong_ignore_zero():
    cl_wise, mean, freq_w = get_iou(np.ones(40), np.ones(40)*2, 3, ignore_zero_class=True)
    assert(
        np.isnan(cl_wise[0]) 
        and np.all(np.equal(cl_wise[1:3],[approx(0),approx(0)]))
        and mean == approx(0)
        and freq_w == approx(0)
    )

def test_7_iou_all_correct_ignore_zero(gt_imbalanced):
    cl_wise, mean, freq_w = get_iou(gt_imbalanced, gt_imbalanced, 3, ignore_zero_class=True)
    assert(
        np.isnan(cl_wise[0]) 
        and np.all(np.equal(cl_wise[1:3],[approx(1),approx(1)]))
        and mean == approx(1)
        and freq_w == approx(1)
    )

def test_7_iou_all_correct(gt_imbalanced):
    cl_wise, mean, freq_w = get_iou(gt_imbalanced, gt_imbalanced, 3, ignore_zero_class=False)
    assert(
        np.all(np.equal(cl_wise,[approx(1),approx(1),approx(1)]))
        and mean == approx(1)
        and freq_w == approx(1)
    )

def test_8_iou_misaligned(gt_with_zero, pr_with_zero_misaligned):
    cl_wise, mean, freq_w = get_iou(gt_with_zero, pr_with_zero_misaligned, 3, ignore_zero_class=False)
    print(cl_wise, mean, freq_w)
    assert(
        np.all(np.equal(cl_wise,[approx(1/3),approx(1/3),approx(1/3)])) 
        and mean == approx(1/3)
        and freq_w == approx(1/3)
        )

def test_9_iou_pr_all1_imbalanced(gt_imbalanced):
    cl_wise, mean, freq_w = get_iou(gt_imbalanced, np.ones(40), 3, ignore_zero_class=False)
    assert(
        np.all(np.equal(cl_wise,[approx(0),approx(1/2),approx(0)]))
        and mean == approx((1/2)/3) 
        and freq_w == approx((1/2)*(1/2))
    )