package com.clust4j.algo;

import static org.junit.Assert.*;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;

import org.junit.Test;

import com.clust4j.TestSuite;
import com.clust4j.except.ModelNotFitException;
import com.clust4j.utils.VecUtils;

public class TestLabelEncoder implements BaseModelTest {

	@Test
	public void test1() {
		int[] labels = new int[]{1,2,2,6};
		LabelEncoder le = new LabelEncoder(labels).fit();
		
		// Test that fit just returns immediately after already fit
		le.fit();
		
		int[] expectedClasses = new int[]{1,2,6};
		assertTrue(VecUtils.equalsExactly(le.getClasses(), expectedClasses));
		assertTrue(le.getNumClasses() == 3);
		assertTrue(le.encodeOrNull(1) == 0);
		
		int[] expectedEncodings = new int[]{0,1,1,2};
		assertTrue(VecUtils.equalsExactly(expectedEncodings, le.getEncodedLabels()));
		assertTrue(VecUtils.equalsExactly(labels, le.getRawLabels()));
		
		assertTrue(le.encodeOrNull(0) == null);
		assertTrue(le.reverseEncodeOrNull(0) == 1);
		
		assertTrue(le.reverseEncodeOrNull(12) == null);
		assertTrue(le.encodeOrNull(2) == 1);
		
		assertTrue(VecUtils.equalsExactly(le.reverseTransform(le.getEncodedLabels()), labels));
		assertTrue(VecUtils.equalsExactly(le.transform(le.getRawLabels()), expectedEncodings));
	}

	@Test(expected=ModelNotFitException.class)
	public void test2() {
		int[] labels = new int[]{1,2,2,6};
		LabelEncoder le = new LabelEncoder(labels);
		le.encodeOrNull(1);
	}
	
	@Test(expected=ModelNotFitException.class)
	public void test3() {
		int[] labels = new int[]{1,2,2,6};
		LabelEncoder le = new LabelEncoder(labels);
		le.reverseEncodeOrNull(1);
	}
	
	@Test(expected=ModelNotFitException.class)
	public void test4() {
		int[] labels = new int[]{1,2,2,6};
		LabelEncoder le = new LabelEncoder(labels);
		le.getEncodedLabels();
	}
	
	@Test(expected=ModelNotFitException.class)
	public void test5() {
		int[] labels = new int[]{1,2,2,6};
		LabelEncoder le = new LabelEncoder(labels);
		le.reverseTransform(new int[]{0,1,1,2});
	}
	
	@Test(expected=ModelNotFitException.class)
	public void test6() {
		int[] labels = new int[]{1,2,2,6};
		LabelEncoder le = new LabelEncoder(labels);
		le.transform(labels);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void test7() {
		int[] labels = new int[]{1,2,2,6};
		LabelEncoder le = new LabelEncoder(labels).fit();
		le.transform(new int[]{7,6,5});
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void test8() {
		int[] labels = new int[]{1,2,2,6};
		LabelEncoder le = new LabelEncoder(labels).fit();
		le.reverseTransform(new int[]{3,2,1});
	}
	
	
	// ====== test unique classes with less than 2
	@Test(expected=IllegalArgumentException.class)
	public void test9() {
		int[] labels = new int[]{};
		new LabelEncoder(labels);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void test10() {
		int[] labels = new int[]{1,1,1,1,1,1,1};
		new LabelEncoder(labels);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void test11() {
		int[] labels = new int[]{1};
		new LabelEncoder(labels);
	}

	@Test
	@Override
	public void testSerialization() throws IOException, ClassNotFoundException {
		LabelEncoder encoder = new LabelEncoder(new int[]{
			0,1,5,1,2,2,2,0,1,1,5
		}).fit();
		
		final int[] mappings = encoder.getEncodedLabels();
		encoder.saveModel(new FileOutputStream(TestSuite.tmpSerPath));
		assertTrue(TestSuite.file.exists());
		
		LabelEncoder encoder2 = (LabelEncoder)LabelEncoder.loadModel(new FileInputStream(TestSuite.tmpSerPath));
		assertTrue(VecUtils.equalsExactly(mappings, encoder2.getEncodedLabels()));
		Files.delete(TestSuite.path);
	}
}
