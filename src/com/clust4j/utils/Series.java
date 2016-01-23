package com.clust4j.utils;

public abstract class Series<T> {
	public static boolean eval(double a, Inequality in, double b) {
		switch(in) {
			case LT:
				return a < b;
			case ET:
				return a == b;
			case GT:
				return a > b;
			case LTOET:
				return a <= b;
			case GTOET:
				return a >= b;
			case NET:
				return a != b;
			default:
				throw new IllegalArgumentException("illegal inequality");
		}
	}
	
	abstract public T get();
	abstract public T getRef();
}
