
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.*;
import weka.classifiers.rules.*;
import weka.classifiers.trees.*;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;

public class Mian {
	
	static Instances completeData = null;
	static Instances data = null;
	static ArrayList<Integer> missingPosition = null;

	public static void main(String... Args){
		// load file
		String filedir = "/run/media/juunnn/JAV/Dataset/data/hypothyroid.csv";
		try {
			System.out.print("Loading data source ....... ");
			loadData(filedir);
			System.out.print("success");
			System.out.println();	
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		
		//separates the missing data and the complete data
		try {
			System.out.print("Spliting data source ....... ");
			splitdata();
			System.out.print("success");
			System.out.println();
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		
		//replace the missing value with new datum
		try {
			System.out.print("Replacing missing values ...... ");	
			ganti();
			System.out.print("success");
			System.out.println();	
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		
		//building the main models
		Classifier model = null;
		try {
			System.out.print("Building model ...... ");
			model = buildModel();
			System.out.println("#==============================================#");
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		
		//evaluation
		try {
			System.out.print("Evaluating model ...... ");
			eval(model);
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		
	}

	private static void eval(Classifier model) throws Exception {
		Evaluation ev = new Evaluation(data);
		ev.crossValidateModel(model, data, 10, new Random(19071996));
		System.out.println(ev.toSummaryString());
		
	}

	private static Classifier buildModel() throws Exception {
		data.setClassIndex(data.numAttributes() - 1);
		J48 model =new J48();
		model.buildClassifier(data);
		System.out.println(model.toString());
		return model;
	}

	private static void ganti() throws Exception {	
		for(Integer pos: missingPosition) {
			Instance ins = data.instance(pos);
			for(int i = 0; i < ins.numAttributes(); i++) {
				if(ins.isMissing(i)) {
					Attribute cls = ins.attribute(i);
					ins.setValue(cls, predict(cls, pos));
				}
			}
		System.out.println(pos);
		}
		
	}

	

	private static double predict(Attribute cls, int pos) throws Exception {
		completeData.setClass(cls);
		data.setClass(cls);;
		Instance sample = data.instance(pos);
		Classifier model = null;
		if(cls.isNumeric()) {
			model = new MultilayerPerceptron();
		}
		else {
			model = new OneR();
		}
		
		
		model.buildClassifier(completeData);
		return model.classifyInstance(sample);
		
	}

	private static void splitdata() throws Exception{
		completeData = new Instances(data, data.numInstances());
		missingPosition = new ArrayList<Integer>();
		for(Instance ins: data) {
			if(!ins.hasMissingValue()) {
				completeData.add(ins);
			}
			else {
				missingPosition.add(data.indexOf(ins));
			}
		}
	}

	private static void loadData(String fileDir) throws IOException {
		File file = new File(fileDir);
		if (file.getName().toLowerCase().endsWith(CSVLoader.FILE_EXTENSION)) {
			CSVLoader loader = new CSVLoader();
			loader.setSource(file);
			 data =  loader.getDataSet();
		} else {
			data =  new Instances(new BufferedReader(new FileReader(file)));
		}
	}

	


}
