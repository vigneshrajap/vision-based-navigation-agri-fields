import pytest
from pytest import approx
import numpy as np
import geometric_utilities

@pytest.fixture
def vector1():
    vector = [1,1]
    return vector/np.linalg.norm(vector)
@pytest.fixture
def vector2():
    vector = [2,1]
    return vector/np.linalg.norm(vector)

@pytest.fixture
def vector1_backwards():
    vector = [-1,-1]
    return vector/np.linalg.norm(vector)

@pytest.fixture
def vector2_backwards():
    vector = [-2,-1]
    return vector/np.linalg.norm(vector)

@pytest.fixture
def vector1_negative():
    vector = [1,-1]
    return vector/np.linalg.norm(vector)

@pytest.fixture
def vector2_negative():
    vector = [2,-1]
    return vector/np.linalg.norm(vector)

@pytest.fixture
def vector1_negative_backwards():
    vector = [-1,1]
    return vector/np.linalg.norm(vector)

@pytest.fixture
def vector2_negative_backwards():
    vector = [-2,1]
    return vector/np.linalg.norm(vector)

@pytest.fixture
def xvector():
    vector = [1,0]
    return vector/np.linalg.norm(vector)

@pytest.fixture
def origo():
    return np.array([0,0])

@pytest.fixture
def point():
    return np.array([2,1])

@pytest.fixture
def point_negative():
    return np.array([2,-1])

@pytest.fixture
def point_backwards():
    return np.array([-2,-1])

@pytest.fixture
def point_xaxis():
    return np.array([1,0])

#Direction tests
def test_direction_both_forwards(vector1, vector2):
    direction_sign = geometric_utilities.direction_sign(vector1,vector2)
    assert direction_sign == 1

def test_direction_both_forwards_one_negative(vector1, vector2_negative):
    direction_sign = geometric_utilities.direction_sign(vector1,vector2_negative)
    assert direction_sign == 1

def test_direction_both_forwards_both_negative(vector1_negative, vector2_negative):
    direction_sign = geometric_utilities.direction_sign(vector1_negative,vector2_negative)
    assert direction_sign == 1

def test_direction_one_backwards(vector1, vector1_backwards):
    direction_sign = geometric_utilities.direction_sign(vector1,vector1_backwards)
    assert direction_sign == -1

def test_direction_both_backwards(vector1_backwards, vector2_backwards):
    direction_sign = geometric_utilities.direction_sign(vector1_backwards,vector2_backwards)
    assert direction_sign == 1

def test_direction_same_vector(vector1):
    direction_sign = geometric_utilities.direction_sign(vector1,vector1)
    assert direction_sign == 1

#Angle
def test_angle_negative_correct_sign(vector1, vector2):
    angle = geometric_utilities.angle_between_vectors(vector1,vector2)
    assert angle < 0

def test_angle_positive_correct_sign(vector1, vector2):
    angle = geometric_utilities.angle_between_vectors(vector2,vector1)
    assert angle > 0

def test_angle_both_negative_correct_sign(vector1_negative, vector2_negative):
    angle = geometric_utilities.angle_between_vectors(vector1_negative,vector2_negative)
    assert angle > 0

def test_angle_positive_negative_correct_sign(vector1, vector2_negative):
    angle = geometric_utilities.angle_between_vectors(vector1,vector2_negative)
    assert angle < 0
def test_angle_negative_correct_value(vector1, xvector):
    angle = geometric_utilities.angle_between_vectors(vector1,xvector)
    assert angle == approx(-np.pi/4)

def test_angle_positive_correct_value(vector1, xvector):
    angle = geometric_utilities.angle_between_vectors(xvector,vector1)
    assert angle == approx(np.pi/4)

#Distance
def test_signed_distance_negative_correct_sign(point, origo, vector1):
    d = geometric_utilities.signed_distance_point_to_line(point=point, line_point=origo, line_vector=vector1)
    assert d < 0

def test_signed_distance_positive_correct_sign(point, origo, xvector):
    d = geometric_utilities.signed_distance_point_to_line(point=point, line_point=origo, line_vector=xvector)
    assert d > 0

def test_signed_distance_both_negative_correct_sign(point_negative, origo, vector1_negative):
    d = geometric_utilities.signed_distance_point_to_line(point=point_negative, line_point=origo, line_vector=vector1_negative)
    assert d > 0

def test_signed_distance_positive_correct_value(point, origo, xvector):
    d = geometric_utilities.signed_distance_point_to_line(point=point, line_point=origo, line_vector=xvector)
    assert d == approx(1)

def test_signed_distance_both_backwards_correct_sign(point_backwards, origo, vector1_backwards):
    d = geometric_utilities.signed_distance_point_to_line(point=point_backwards, line_point=origo, line_vector=vector1_backwards)
    assert d < 0

def test_closest_point(point,point_negative,point_xaxis,origo):
    ind1 = geometric_utilities.closest_point(point[0],point[1],[point_xaxis[0],origo[0]],[point_xaxis[1],origo[1]])
    ind2 = geometric_utilities.closest_point(point_negative[0],point_negative[1],[point_xaxis[0],origo[0]],[point_xaxis[1],origo[1]])
    ind3 = geometric_utilities.closest_point(origo[0],origo[1],[point_xaxis[0],point[0], point_negative[0]],[point_xaxis[1],point[1], point_negative[1]])
    assert ind1 == 0
    assert ind2 == 0
    assert ind3 == 0



