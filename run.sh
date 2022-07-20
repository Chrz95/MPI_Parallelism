# ./run.sh

#####################################
echo   " "
echo   "Executing Reference Program : "
gcc -o exec reference_code.c
./exec 10000000

rm exec

echo   " "
echo   "Executing SSE Program : "
gcc -o exec SSE_code.c -msse4.2
./exec 10000000

rm exec

echo   "Executing SSE (Bonus) Program : "
gcc -o exec SSE_code_Bonus.c -msse4.2
./exec 10000000

rm exec

echo   "Executing SSE + Pthread Program (2 Threads) : "
gcc -o exec SSE_PTH_code.c -msse4.2 -lpthread
./exec 10000000 2

rm exec

echo   " "
echo   "Executing SSE + Pthread Program (4 Threads): "
gcc -o exec SSE_PTH_code.c -msse4.2 -lpthread
./exec 10000000 4

rm exec

echo   " "
echo   "Executing SSE + Pthread + MPI Program (Threads,Processes) = (2,2) : "
mpicc -o exec SSE_PTH_MPI_code.c -msse4.2 -lpthread
lamboot -v
mpiexec -n 2 ./exec 10000000 2 

rm exec

echo   " "
echo   "Executing SSE + Pthread + MPI Program (Threads,Processes) = (2,4) : "
mpicc -o exec SSE_PTH_MPI_code.c -msse4.2 -lpthread
lamboot -v
mpiexec -n 2 ./exec 10000000 4 

rm exec

echo   " "
echo   "Executing SSE + Pthread + MPI Program (Threads,Processes) = (4,2): "
mpicc -o exec SSE_PTH_MPI_code.c -msse4.2 -lpthread
lamboot -v
mpiexec -n 4 ./exec 10000000 2 

rm exec

echo   " "
echo   "Executing SSE + Pthread + MPI Program (Threads,Processes) = (4,4) : "
mpicc -o exec SSE_PTH_MPI_code.c -msse4.2 -lpthread
lamboot -v
mpiexec -n 4 ./exec 10000000 4 

rm exec