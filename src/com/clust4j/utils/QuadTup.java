package com.clust4j.utils;

public class QuadTup<C_ONE, C_TWO, C_THREE, C_FOUR> {
	public final C_ONE one;
	public final C_TWO two;
	public final C_THREE three;
	public final C_FOUR four;
	
	public QuadTup(C_ONE one, C_TWO two, C_THREE three, C_FOUR four) {
		this.one = one;
		this.two = two;
		this.three = three;
		this.four = four;
	}
	
	@Override public String toString() {
		return "("+one+", "+two+", "+three+", "+four+")";
	}
}
