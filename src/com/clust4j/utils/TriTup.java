package com.clust4j.utils;

public class TriTup<C_ONE, C_TWO, C_THREE> {
	public final C_ONE one;
	public final C_TWO two;
	public final C_THREE three;
	
	public TriTup(C_ONE one, C_TWO two, C_THREE three) {
		this.one = one;
		this.two = two;
		this.three = three;
	}
	
	@Override public String toString() {
		return "("+one+", "+two+", "+three+")";
	}
}