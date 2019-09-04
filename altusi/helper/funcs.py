import argparse

def getArgs():
    """Argument collecting and parsing

    Collect and parse user arguments for program

    Returns:
    --------
        args : argparse object 
            arguments after parsing
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--video', '-v', type=str,
                        required=False,
                        help='path to video stream')
    parser.add_argument('--image', '-i', type=str,
                        required=False,
                        help='path to image file')
    parser.add_argument('--flip_hor', '-fh',
                        default=False, required=False,
                        action='store_true',
                        help='horizontally flip video')
    parser.add_argument('--flip_ver', '-fv',
                        default=False, required=False,
                        action='store_true',
                        help='vertically flip video')
    parser.add_argument('--show', '-s', 
                        default=False, required=False,
                        action='store_true',
                        help='whether or not the output is visualized')
    parser.add_argument('--record', '-r',
                        default=False, required=False,
                        action='store_true',
                        help='whether output video is saved')
    parser.add_argument('--name', '-n', type=str,
                        default='camera', required=False,
                        help='name of video stream used for recording')

    args = parser.parse_args()

    return args 
