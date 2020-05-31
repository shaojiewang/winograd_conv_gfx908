import wino_conv_param
import argparse
from sympy import Rational

# np version points
def _interpolation_points(degree):
    """Propose filter points"""

    assert 2 < degree < 18

    # Default interpolation lookup table
    #
    # [1] Error Analysis and Improving the Accuracy of Winograd Convolution for Deep Neural Networks
    #     Barbara Barabasz, Andrew Anderson, Kirk M. Soodhalter, David Gregg
    #     https://arxiv.org/abs/1803.10986
    #

    # pylint: disable=bad-whitespace,line-too-long
    in_pts = [
        #   {invalid}
        [],
        #01 {E=4.63E-08 on conv2d  [1]}
        [],
        #02 {E=7.65E-08 on F( 2,3) [1]}
        [0,   -1,    1],
        #03 {E=2.35E-07 on F( 3,3) [1]}
        [0,   -1,    1,  1/2],
        #04 {E=3.29E-07 on F( 4,3) [1]}
        [0,   -1,    1,  1/2,   -2],
        #05 {E=6.81E-07 on F( 5,3) [1]}
        [0,   -1,    1,  1/2,   -2, -1/2],
        #06 {E=8.79E-07 on F( 6,3) [1]}
        [0,   -1,    1,  1/2, -1/2,    2,   -2],
        #07 {E=3.71E-06 on F( 7,3) [1]}
        [0,   -1,    1,  1/2, -1/2,    2,   -2, -1/4],
        #08 {E=7.35E-06 on F( 8,3) [1]}
        [0,   -1,    1,  1/2, -1/2,    2,   -2, -1/4,    4],
        #09 {E=2.20E-05 on F( 9,3) [1]}
        [0,   -1,    1,  1/2, -1/2,    2,   -2, -1/4,  3/4, -4/3],
        #10 {E=3.22E-05 on F(10,3) [1]}
        [0,   -1,    1,  1/2, -1/2,    2,   -2, -1/4,    4,  3/4, -4/3],
        #11 {E=1.09E-04 on F(11,3) [1]}
        [0,   -1,    1,  1/2, -1/2,    2,   -2, -1/4,    4,  3/4, -4/3,  1/4],
        #12 {E=1.99E-04 on F(12,3) [1]}
        [0,   -1,    1,  1/2, -1/2,    2,   -2, -1/4,    4,  1/4, -3/4,  4/3,   -4],
        #13 {E=5.54E-04 on F(13,3) [1]}
        [0,   -1,    1,  1/2, -1/2,    2,   -2, -1/4,    4,  1/4, -3/4,  4/3,  3/4, -4/3],
        #14 {E=8.80E-04 on F(14,3) [1]}
        [0,   -1,    1,  1/2, -1/2,    2,   -2, -1/4,    4,  1/4, -3/4,  4/3,   -4,  3/4, -4/3],
        #15 {E=1.07E-02 on F(15,3) [1]}
        [0,   -1,    1,  1/2, -1/2,    2,   -2, -1/4,    4,  1/4, -3/4,  4/3,   -4,  2/3, -3/2,  3/2],
        #16 {E=1.93E-02 on F(16,3) [1]}
        [0,   -1,    1,  1/2, -1/2,    2,   -2, -1/4,    4,  1/4, -3/4,  4/3,   -4,  2/3, -3/2, -2/3,  3/2]
    ] # pylint: enable=bad-whitespace,line-too-long

    #return in_pts[degree-1]
    return np.array(in_pts[degree-1], dtype=np.float64)

# matrix version points
def interpolation_points(degree):
    """Propose filter points"""

    assert 2 < degree < 18

    # Default interpolation lookup table
    #
    # [1] Error Analysis and Improving the Accuracy of Winograd Convolution for Deep Neural Networks
    #     Barbara Barabasz, Andrew Anderson, Kirk M. Soodhalter, David Gregg
    #     https://arxiv.org/abs/1803.10986
    #

    # pylint: disable=bad-whitespace,line-too-long
    in_pts = [
        #   {invalid}
        [],
        #01 {E=4.63E-08 on conv2d  [1]}
        [],
        #02 {E=7.65E-08 on F( 2,3) [1]}
        [0,   -1,    1],
        #03 {E=2.35E-07 on F( 3,3) [1]}
        [0,   -1,    1,  Rational(1/2)],
        #04 {E=3.29E-07 on F( 4,3) [1]}
        [0,   -1,    1,  Rational(1/2),   -2],
        #05 {E=6.81E-07 on F( 5,3) [1]}
        [0,   -1,    1,  Rational(1/2),   -2, -Rational(1/2)],
        #06 {E=8.79E-07 on F( 6,3) [1]}
        [0,   -1,    1,  Rational(1/2), -Rational(1/2),    2,   -2],
        #07 {E=3.71E-06 on F( 7,3) [1]}
        [0,   -1,    1,  Rational(1/2), -Rational(1/2),    2,   -2, -Rational(1/4)],
        #08 {E=7.35E-06 on F( 8,3) [1]}
        [0,   -1,    1,  Rational(1/2), -Rational(1/2),    2,   -2, -Rational(1/4),    4],
        #09 {E=2.20E-05 on F( 9,3) [1]}
        [0,   -1,    1,  Rational(1/2), -Rational(1/2),    2,   -2, -Rational(1/4),  Rational(3/4), -Rational(4/3)],
        #10 {E=3.22E-05 on F(10,3) [1]}
        [0,   -1,    1,  Rational(1/2), -Rational(1/2),    2,   -2, -Rational(1/4),    4,  Rational(3/4), -Rational(4/3)],
        #11 {E=1.09E-04 on F(11,3) [1]}
        [0,   -1,    1,  Rational(1/2), -Rational(1/2),    2,   -2, -Rational(1/4),    4,  Rational(3/4), -Rational(4/3),  Rational(1/4)],
        #12 {E=1.99E-04 on F(12,3) [1]}
        [0,   -1,    1,  Rational(1/2), -Rational(1/2),    2,   -2, -Rational(1/4),    4,  Rational(1/4), -Rational(3/4),  Rational(4/3),   -4],
        #13 {E=5.54E-04 on F(13,3) [1]}
        [0,   -1,    1,  Rational(1/2), -Rational(1/2),    2,   -2, -Rational(1/4),    4,  Rational(1/4), -Rational(3/4),  Rational(4/3),  Rational(3/4), -Rational(4/3)],
        #14 {E=8.80E-04 on F(14,3) [1]}
        [0,   -1,    1,  Rational(1/2), -Rational(1/2),    2,   -2, -Rational(1/4),    4,  Rational(1/4), -Rational(3/4),  Rational(4/3),   -4,  Rational(3/4), -Rational(4/3)],
        #15 {E=1.07E-02 on F(15,3) [1]}
        [0,   -1,    1,  Rational(1/2), -Rational(1/2),    2,   -2, -Rational(1/4),    4,  Rational(1/4), -Rational(3/4),  Rational(4/3),   -4,  Rational(2/3), -Rational(3/2),  Rational(3/2)],
        #16 {E=1.93E-02 on F(16,3) [1]}
        [0,   -1,    1,  Rational(1/2), -Rational(1/2),    2,   -2, -Rational(1/4),    4,  Rational(1/4), -Rational(3/4),  Rational(4/3),   -4,  Rational(2/3), -Rational(3/2), -Rational(2/3),  Rational(3/2)]
    ] # pylint: enable=bad-whitespace,line-too-long

    return in_pts[degree-1]

def winograd_transform_matrices(tile_size, kernel_size):
    """Compute the A, B, and G transform matrices for `tile_size` as a `tvm.Expr`.
    """
    if not 1 < tile_size < 9:
        raise ValueError("Unsupported tile size for Winograd: {}".format(tile_size))
    if not 2 < kernel_size < 8:
        raise ValueError("Unsupported kernel size for Winograd: {}".format(kernel_size))

    degree = tile_size + kernel_size - 2

    intp_pts = interpolation_points(degree)
    wino_conv_param.showCookToomConvolution(intp_pts, tile_size, kernel_size)
    #A_data, B_data, G_data = wino_conv_param.showCookToomConvolution(intp_pts, tile_size, kernel_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tile_size", help="tile size n")
    parser.add_argument("filter_size", help="filter size f")

    args = parser.parse_args()
    n = args.tile_size
    f = args.filter_size
    winograd_transform_matrices(int(n), int(f))