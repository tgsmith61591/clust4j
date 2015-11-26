package com.clust4j.viz;

import org.apache.commons.math3.exception.DimensionMismatchException;
import com.clust4j.utils.VecUtils;

public class Axis2DVizualizer {
	private final ClassLabelColorizer colors;
	private final double[] x;
	private final double[] y;
	private final int m;
	
	public Axis2DVizualizer(final double[] x, final double[] y, final int[] labels) {
		if(labels.length != (m=x.length))
			throw new DimensionMismatchException(labels.length, m);
		if(m != y.length)
			throw new DimensionMismatchException(m, y.length);
		
		this.x = VecUtils.copy(x);
		this.y = VecUtils.copy(y);
		
		colors = new ClassLabelColorizer(labels);
	}
	
	public void draw() {
		// TODO
	}
	
	public ClassLabelColorizer getClassLabelColors() {
		return colors;
	}
	
	public double[] getXCopy() {
		return VecUtils.copy(x);
	}
	
	public double[] getYCopy() {
		return VecUtils.copy(y);
	}
}
