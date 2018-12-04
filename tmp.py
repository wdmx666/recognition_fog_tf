
x = [11,4,5,40, 82, 97, 47, 7]
def partition(arr,low,high):
    i = ( low-1 )         # index of smaller element
    pivot = arr[high]     # pivot

    for j in range(low , high):
        # If current element is smaller than or equal to pivot
        if arr[j] <= pivot:
            # increment index of smaller element
            i = i+1
            print(f'{i,j,arr}')
            tmp_1 = arr[i]
            tmp_2 = arr[j]
            arr[i], arr[j] = arr[j],arr[i]

    arr[i+1], arr[high] = arr[high],arr[i+1]
    return ( i+1 )
partition(x,0,7)