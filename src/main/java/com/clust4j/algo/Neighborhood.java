package com.clust4j.algo;

import com.clust4j.utils.DeepCloneable;
import com.clust4j.utils.EntryPair;
import com.clust4j.utils.MatUtils;
import com.clust4j.utils.MatrixFormatter;

public class Neighborhood 
		extends EntryPair<double[][], int[][]> 
		implements DeepCloneable, java.io.Serializable {

	private static final long serialVersionUID = 2016176782694689004L;
	private static final MatrixFormatter fmt = new MatrixFormatter();

	public Neighborhood(EntryPair<double[][], int[][]> entry) {
		this(entry.getKey(), entry.getValue());
	}

	public Neighborhood(double[][] key, int[][] value) {
		super(key, value);
	}

	@Override
	public Neighborhood copy() {
		return new Neighborhood(MatUtils.copy(getDistances()), MatUtils.copy(getIndices()));
	}

	@Override
	public boolean equals(Object o) {
		if (this == o)
			return true;
		if (o instanceof Neighborhood) {
			Neighborhood n = (Neighborhood) o;
			return MatUtils.equalsExactly(this.getKey(), n.getKey())
					&& MatUtils.equalsExactly(this.getValue(), n.getValue());
		}

		return false;
	}

	public double[][] getDistances() {
		return getKey();
	}

	public int[][] getIndices() {
		return getValue();
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		String ls = System.getProperty("line.separator");

		sb.append("Distances:" + ls + fmt.format(getDistances()) + ls + ls);
		sb.append("Indices:" + ls + fmt.format(getIndices()));

		return sb.toString();
	}
}