def check_is_Eucledian_distance(pos):
    n = len(pos)
    D = DistanceMatrix(pos).distance_matrix()

    for i in range(n):
        for j in range(i+1, n):
            for k in range(n):
                if (k!=i) and (k!=j):
                    if D[i][j]+D[i][k]>D[j][k]:
                        return False
    return True