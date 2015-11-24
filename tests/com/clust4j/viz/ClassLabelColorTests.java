package com.clust4j.viz;

import static org.junit.Assert.*;

import java.awt.Color;

import org.junit.Test;

import com.clust4j.utils.VecUtils;

public class ClassLabelColorTests {

	@Test
	public void test() {
		for(int i = 0; i < 1000; i++) { // Perform many times to simulate the random seed
			final int[] labels = new int[]{-1,1,0,-1,0,1,0,1,0,-1};
			final Color[] colors = new ClassLabelColorizer(labels).toColorArray();
			assertTrue(VecUtils.unique(colors).size() == 3);
		}
	}

}
