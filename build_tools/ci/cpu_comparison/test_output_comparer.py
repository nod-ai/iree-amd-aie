from .output_comparer import compare, Summarizer
import numpy as np


def stripWhitespace(s):
    """
    Strip all whitespace from the end of each line of a string, and remove all
    empty lines at the start and end
    """
    s = s.strip("\n").strip(" ")
    lines = s.split("\n")
    lines = [line.rstrip() for line in lines]
    s = "\n".join(lines)
    return s


def test_compare_0():
    """ """

    A = np.arange(20 * 30 * 40).reshape(20, 30, 40)
    A = np.array(A, dtype=np.float32)
    B = A.copy()

    B[10, 3, 4] = 17
    B[10, 3, 5] = 17
    B[10, 3, 6] = 17
    B[10, 2, 4] = 22
    B[10, 2, 5] = 99999
    B[10, 2, 6] = 22

    summarizer = Summarizer(A, B, 0, 0)

    expected = """
Values are not all close, here is a summary of the differences:

- number of positions where values are different is 6 out of 24000
- maximum absolute difference: 87914.0
- mean absolute difference: 6.181833267211914
- number of zeros in baseline array: 1
- number of zeros in AIE array: 1
- baseline values are in the range [0.0, 23999.0]
- AIE values are in the range [0.0, 99999.0]
"""
    a = stripWhitespace(summarizer.getBaseString())
    b = stripWhitespace(expected)

    assert a == b, f"Strings do not match."

    expected = """
Discrepencies at first 6 indices:

 Index       | Baseline                | AIE
-------------+-------------------------+----
[[10  2  4]  | 12084.0                 | 22.0
 [10  2  5]  | 12085.0                 | 99999.0
 [10  2  6]  | 12086.0                 | 22.0
 [10  3  4]  | 12124.0                 | 17.0
 [10  3  5]  | 12125.0                 | 17.0
 [10  3  6]] | 12126.0                 | 17.0
-------------+-------------------------+----
"""

    a = stripWhitespace(summarizer.getDiscrepenciesString(40))
    b = stripWhitespace(expected)

    assert a == b, f"Strings do not match."

    expected = """
Baseline result in the slice range [7:13,0:5,1:7]
[[[ 8401  8402  8403  8404  8405  8406]
  [ 8441  8442  8443  8444  8445  8446]
  [ 8481  8482  8483  8484  8485  8486]
  [ 8521  8522  8523  8524  8525  8526]
  [ 8561  8562  8563  8564  8565  8566]]

 [[ 9601  9602  9603  9604  9605  9606]
  [ 9641  9642  9643  9644  9645  9646]
  [ 9681  9682  9683  9684  9685  9686]
  [ 9721  9722  9723  9724  9725  9726]
  [ 9761  9762  9763  9764  9765  9766]]

 [[10801 10802 10803 10804 10805 10806]
  [10841 10842 10843 10844 10845 10846]
  [10881 10882 10883 10884 10885 10886]
  [10921 10922 10923 10924 10925 10926]
  [10961 10962 10963 10964 10965 10966]]

 [[12001 12002 12003 12004 12005 12006]
  [12041 12042 12043 12044 12045 12046]
  [12081 12082 12083 12084 12085 12086]
  [12121 12122 12123 12124 12125 12126]
  [12161 12162 12163 12164 12165 12166]]

 [[13201 13202 13203 13204 13205 13206]
  [13241 13242 13243 13244 13245 13246]
  [13281 13282 13283 13284 13285 13286]
  [13321 13322 13323 13324 13325 13326]
  [13361 13362 13363 13364 13365 13366]]

 [[14401 14402 14403 14404 14405 14406]
  [14441 14442 14443 14444 14445 14446]
  [14481 14482 14483 14484 14485 14486]
  [14521 14522 14523 14524 14525 14526]
  [14561 14562 14563 14564 14565 14566]]]


AIE result in the slice range [7:13,0:5,1:7]
[[[ 8401  8402  8403  8404  8405  8406]
  [ 8441  8442  8443  8444  8445  8446]
  [ 8481  8482  8483  8484  8485  8486]
  [ 8521  8522  8523  8524  8525  8526]
  [ 8561  8562  8563  8564  8565  8566]]

 [[ 9601  9602  9603  9604  9605  9606]
  [ 9641  9642  9643  9644  9645  9646]
  [ 9681  9682  9683  9684  9685  9686]
  [ 9721  9722  9723  9724  9725  9726]
  [ 9761  9762  9763  9764  9765  9766]]

 [[10801 10802 10803 10804 10805 10806]
  [10841 10842 10843 10844 10845 10846]
  [10881 10882 10883 10884 10885 10886]
  [10921 10922 10923 10924 10925 10926]
  [10961 10962 10963 10964 10965 10966]]

 [[12001 12002 12003 12004 12005 12006]
  [12041 12042 12043 12044 12045 12046]
  [12081 12082 12083    22 99999    22]
  [12121 12122 12123    17    17    17]
  [12161 12162 12163 12164 12165 12166]]

 [[13201 13202 13203 13204 13205 13206]
  [13241 13242 13243 13244 13245 13246]
  [13281 13282 13283 13284 13285 13286]
  [13321 13322 13323 13324 13325 13326]
  [13361 13362 13363 13364 13365 13366]]

 [[14401 14402 14403 14404 14405 14406]
  [14441 14442 14443 14444 14445 14446]
  [14481 14482 14483 14484 14485 14486]
  [14521 14522 14523 14524 14525 14526]
  [14561 14562 14563 14564 14565 14566]]]


Absolute difference in the slice range [7:13,0:5,1:7]
[[[    0     0     0     0     0     0]
  [    0     0     0     0     0     0]
  [    0     0     0     0     0     0]
  [    0     0     0     0     0     0]
  [    0     0     0     0     0     0]]

 [[    0     0     0     0     0     0]
  [    0     0     0     0     0     0]
  [    0     0     0     0     0     0]
  [    0     0     0     0     0     0]
  [    0     0     0     0     0     0]]

 [[    0     0     0     0     0     0]
  [    0     0     0     0     0     0]
  [    0     0     0     0     0     0]
  [    0     0     0     0     0     0]
  [    0     0     0     0     0     0]]

 [[    0     0     0     0     0     0]
  [    0     0     0     0     0     0]
  [    0     0     0 12062 87914 12064]
  [    0     0     0 12107 12108 12109]
  [    0     0     0     0     0     0]]

 [[    0     0     0     0     0     0]
  [    0     0     0     0     0     0]
  [    0     0     0     0     0     0]
  [    0     0     0     0     0     0]
  [    0     0     0     0     0     0]]

 [[    0     0     0     0     0     0]
  [    0     0     0     0     0     0]
  [    0     0     0     0     0     0]
  [    0     0     0     0     0     0]
  [    0     0     0     0     0     0]]]
"""

    a = stripWhitespace(summarizer.getSlicesString())
    b = stripWhitespace(expected)
    assert a == b


def test_compare_1():
    """ """

    A = np.arange(5 * 6).reshape(5, 6)
    B = A.copy()
    B[2, 3] = 90
    B[3, 4] = 0

    expected = """
Values are not all close, here is a summary of the differences:

- number of positions where values are different is 2 out of 30
- maximum absolute difference: 75
- mean absolute difference: 3.2333333333333334
- number of zeros in baseline array: 1
- number of zeros in AIE array: 2
- baseline values are in the range [0, 29]
- AIE values are in the range [0, 90]


Discrepencies at first 2 indices:

 Index  | Baseline                | AIE
--------+-------------------------+----
[[2 3]  | 15                      | 90
 [3 4]] | 22                      | 0
--------+-------------------------+----


Baseline result in the slice range [0:5,0:6]
[[ 0  1  2  3  4  5]
 [ 6  7  8  9 10 11]
 [12 13 14 15 16 17]
 [18 19 20 21 22 23]
 [24 25 26 27 28 29]]


AIE result in the slice range [0:5,0:6]
[[ 0  1  2  3  4  5]
 [ 6  7  8  9 10 11]
 [12 13 14 90 16 17]
 [18 19 20 21  0 23]
 [24 25 26 27 28 29]]


Absolute difference in the slice range [0:5,0:6]
[[ 0  0  0  0  0  0]
 [ 0  0  0  0  0  0]
 [ 0  0  0 75  0  0]
 [ 0  0  0  0 22  0]
 [ 0  0  0  0  0  0]]
"""
    a = stripWhitespace(compare(A, B, 0, 0))
    b = stripWhitespace(expected)

    assert a == b, f"Strings do not match."


def test_compare_2():
    A = np.arange(2 * 2 * 50).reshape(2, 2, 50)
    B = A.copy()
    B[0, 1, 48] = -13

    expected = """
Values are not all close, here is a summary of the differences:

- number of positions where values are different is 1 out of 200
- maximum absolute difference: 111
- mean absolute difference: 0.555
- number of zeros in baseline array: 1
- number of zeros in AIE array: 1
- baseline values are in the range [0, 199]
- AIE values are in the range [-13, 199]


Discrepencies at first 1 indices:

 Index       | Baseline                | AIE
-------------+-------------------------+----
[[ 0  1 48]] | 98                      | -13
-------------+-------------------------+----


Baseline result in the slice range [0:2,0:2,45:50]
[[[ 45  46  47  48  49]
  [ 95  96  97  98  99]]

 [[145 146 147 148 149]
  [195 196 197 198 199]]]


AIE result in the slice range [0:2,0:2,45:50]
[[[ 45  46  47  48  49]
  [ 95  96  97 -13  99]]

 [[145 146 147 148 149]
  [195 196 197 198 199]]]


Absolute difference in the slice range [0:2,0:2,45:50]
[[[  0   0   0   0   0]
  [  0   0   0 111   0]]

 [[  0   0   0   0   0]
  [  0   0   0   0   0]]]
"""

    a = stripWhitespace(compare(A, B, 0, 0))
    b = stripWhitespace(expected)

    assert a == b, f"Strings do not match."


def test_compare_equal():
    A = np.arange(2 * 2 * 50).reshape(2, 2, 50)
    B = A.copy()
    expected = ""
    assert compare(A, B, 0, 0) == expected, f"Strings do not match."
