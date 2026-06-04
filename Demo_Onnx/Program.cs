using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Demo_Onnx
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello, World!");

            // 1、指定ONNX模型路径
            string modelPath = @"C:\Temp\train_model.onnx";

            // 2、加载模型
            var options = new SessionOptions();
            // CPU :运行模型
            // GPU :独立显卡
            options.AppendExecutionProvider_OpenVINO("CPU");
            using (var onnxSession = new InferenceSession(modelPath, options))
            {
                // 3、开始预测
                var inputName = onnxSession.InputNames[0];
                var outputName = onnxSession.OutputNames[0];
                Console.WriteLine($"输入节点名称：{inputName}，输出节点名称：{outputName}");

                // 3.1、预测数据
                // 1、构造onnx模型需要的 输入矩阵数据
                float[] testData = { 1.0f, 1.0f, 0.0f }; // 一维数组
                float[] testData1 = { 0.0f, 1.0f, 0.0f }; // 一维数组

                Predict(onnxSession, testData);
                Predict(onnxSession, testData1);

            }

            Console.WriteLine("Bye, World!");
        }

        static void Predict(InferenceSession onnxSession, float[] testData)
        {
            var inputTensor = new DenseTensor<float>(testData, new int[] { 1, 3 }); // 构造输入张量，形状为 [1, 3]
            var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor(onnxSession.InputNames[0], inputTensor)
                };

            // 2、执行预测
            using (var result = onnxSession.Run(inputs))
            {
                var pre_result = result[0].AsTensor<float>()[0];
                Console.WriteLine($"[1,1,0]预测概率:{pre_result:F6}");
            }
        }
    }
}
