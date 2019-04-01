"""
This problem was asked by Google.

Suppose we represent our file system by a string in the following manner:

The string "dir\n\tsubdir1\n\tsubdir2\n\t\tfile.ext" represents:

dir
    subdir1
    subdir2
        file.ext
The directory dir contains an empty sub-directory subdir1 and a sub-directory subdir2 containing a file file.ext.

The string "dir\n\tsubdir1\n\t\tfile1.ext\n\t\tsubsubdir1\n\tsubdir2\n\t\tsubsubdir2\n\t\t\tfile2.ext" represents:

dir
    subdir1
        file1.ext
        subsubdir1
    subdir2
        subsubdir2
            file2.ext
The directory dir contains two sub-directories subdir1 and subdir2. subdir1 contains a file file1.ext and an empty second-level sub-directory subsubdir1. subdir2 contains a second-level sub-directory subsubdir2 containing a file file2.ext.

We are interested in finding the longest (number of characters) absolute path to a file within our file system. For example, in the second example above, the longest absolute path is "dir/subdir2/subsubdir2/file2.ext", and its length is 32 (not including the double quotes).

Given a string representing the file system in the above format, return the length of the longest absolute path to a file in the abstracted file system. If there is no file in the system, return 0.

Note:

The name of a file contains at least a period and an extension.

The name of a directory or sub-directory will not contain a period.
"""

def longest_abs_path(in_path):
    chunks = in_path.split('\n')

    cur_path_prefix_lens = []
    cur_path_prefix = []
    longest_abs_path = []
    longest_abs_path_len = -1

    for f_name in chunks:
        depth = 0
        while f_name[depth] == '\t':
            depth += 1
        if depth < len(cur_path_prefix):
            cur_path_prefix = cur_path_prefix[:depth]
            cur_path_prefix_lens = cur_path_prefix_lens[:depth]

        cur_path_prefix.append(f_name.strip())
        cur_path_prefix_lens.append(len(f_name.strip()) + 1)

        if '.' in f_name:
            if sum(cur_path_prefix_lens) > longest_abs_path_len:
                longest_abs_path = '/'.join(cur_path_prefix)
                longest_abs_path_len = sum(cur_path_prefix_lens)

    return longest_abs_path


if __name__ == '__main__':
    in_str = 'dir\n\tsubdir1\n\t\tfile1.ext\n\t\tsubsubdir1\n\tsubdir2\n\t\tsubsubdir2\n\t\t\tfile2.ext'

    print(longest_abs_path(in_str))
