# Set the output terminal, for example, to a PNG image
set terminal qt

# Set the output file name
set output 'benchmark_embeddings.png'

# Set labels
set xlabel 'Query length, characters'
set ylabel 'Threads'
set zlabel 'Latency, ms' offset -5,0,0

# Set the title
set title 'Embedding Benchmark'

# Set tics level for better 3D effect
set ticslevel 0.5

# Optional: Set grid
set grid

# Optional: Set the palette if you want colored lines
set palette defined (0 "blue", 1 "red")

# Loop through each file and plot the data
# Assuming files are named 1.out, 2.out, ...
splot for [i=1:20] sprintf('%d.out', i) using 1:(i):2 with linespoints notitle
