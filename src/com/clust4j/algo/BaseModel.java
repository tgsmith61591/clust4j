package com.clust4j.algo;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

abstract public class BaseModel implements java.io.Serializable {
	private static final long serialVersionUID = 4707757741169405063L;

	/** This should be synchronized and thread-safe */
	abstract BaseModel fit();
	
	/**
	 * Load a model from a FileInputStream
	 * @param fos
	 * @return
	 * @throws IOException
	 * @throws ClassNotFoundException
	 */
	public static BaseModel loadModel(final FileInputStream fis) throws IOException, ClassNotFoundException {
		ObjectInputStream in = new ObjectInputStream(fis);
        BaseModel bm = (BaseModel) in.readObject();
        in.close();
        fis.close();
        
        return bm;
	}
	
	/**
	 * Save a model to FileOutputStream
	 * @param fos
	 * @throws IOException
	 */
	public void saveModel(final FileOutputStream fos) throws IOException {
		ObjectOutputStream out = new ObjectOutputStream(fos);
		out.writeObject(this);
		out.close();
		fos.close();
	}
}
