package com.clust4j.utils;

import java.util.Arrays;

public class ArrayFormatter {
	public static String arrayToString(byte[] d) {
		if(null == d)
			return null;
		return arrayToStringFromCA(Arrays.toString(d).toCharArray());
	}
	
	public static String arrayToString(short[] d) {
		if(null == d)
			return null;
		return arrayToStringFromCA(Arrays.toString(d).toCharArray());
	}
	
	public static String arrayToString(int[] d) {
		if(null == d)
			return null;
		return arrayToStringFromCA(Arrays.toString(d).toCharArray());
	}
	
	public static String arrayToString(boolean[] d) {
		if(null == d)
			return null;
		return arrayToStringFromCA(Arrays.toString(d).toCharArray());
	}
	
	public static String arrayToString(float[] d) {
		if(null == d)
			return null;
		return arrayToStringFromCA(Arrays.toString(d).toCharArray());
	}
	
	public static String arrayToString(double[] d) {
		if(null == d)
			return null;
		return arrayToStringFromCA(Arrays.toString(d).toCharArray());
	}
	
	public static String arrayToString(long[] d) {
		if(null == d)
			return null;
		return arrayToStringFromCA(Arrays.toString(d).toCharArray());
	}
	
	public static String arrayToString(String[] d) {
		if(null == d)
			return null;
		return arrayToStringFromCA(Arrays.toString(d).toCharArray());
	}
	
	public static String arrayToString(char[] d) {
		if(null == d)
			return null;
		return arrayToStringFromCA(Arrays.toString(d).toCharArray());
	}
	
	private static String arrayToStringFromCA(final char[] c) {
		final int padding_chars= 6;
		final int max_len = 2 * padding_chars + 3;
		
		if(c.length > max_len) {
			StringBuilder sb = new StringBuilder();
			
			int ws = 0, next_pos = 0;
			while(padding_chars > ws++) {
				sb.append(c[next_pos++]);
			}
			
			sb.append("..."); // le ellipsis
			
			ws = 0;
			next_pos = c.length - padding_chars;
			while(padding_chars > ws++) {
				sb.append(c[next_pos++]);
			}
			
			return sb.toString();
		}
		
		return new String(c);
	}
}
