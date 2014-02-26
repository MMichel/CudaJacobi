import sys
from random import randrange


def gen_matrix(size):
    matrix = []
    for i in xrange(size):
        row = []
        for j in xrange(size):
            rand_num = randrange(1,size+1)
            row.append(rand_num)
        # ensure diagonal dominance here:
        row[i] = sum(row) + 1
        matrix.append(row)
    return matrix


def get_random_solution(size):
    solution = []
    for i in xrange(size):
        rand_num = randrange(1,size+1)
        solution.append(rand_num)
    return solution


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print "Usage: python gen_diag_dominant_matrix.py <size> <output_filename>\n"
        sys.exit()

    size = int(sys.argv[1])
    fname = sys.argv[2]

    matrix = gen_matrix(size)
    solution = get_random_solution(size)

    outfile = open(fname, 'w')

    for row in matrix:
        outfile.write('\n'.join(map(str,row)))
        outfile.write('\n')
    outfile.write('\n'.join(map(str,solution)))
    outfile.close
