#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/float32.hpp"



using namespace std::placeholders;

class MinimalDepthSubscriber : public rclcpp::Node
{
public:
  MinimalDepthSubscriber()
  : Node("depth_sensing")
  {
    /* Note: it is very important to use a QOS profile for the subscriber that is compatible
         * with the QOS profile of the publisher.
         * The ZED component node uses a default QoS profile with reliability set as "RELIABLE"
         * and durability set as "VOLATILE".
         * To be able to receive the subscribed topic the subscriber must use compatible
         * parameters.
         */

    // https://github.com/ros2/ros2/wiki/About-Quality-of-Service-Settings

    rclcpp::QoS depth_qos(10);
    depth_qos.keep_last(10);
    depth_qos.best_effort();
    depth_qos.durability_volatile();

    // Create depth map subscriber
    mDepthSub = create_subscription<sensor_msgs::msg::Image>(
      "/zed/zed_node/depth/depth_registered", depth_qos, std::bind(&MinimalDepthSubscriber::depthCallback, this, _1));

    // Create center distance publisher
    mDepthPub = create_publisher<std_msgs::msg::Float32>("depth_info", 10);
  }

protected:
  void depthCallback(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    // Get a pointer to the depth values casting the data
    // pointer to floating point
    float * depths = reinterpret_cast<float *>(&msg->data[0]);

    // Image coordinates of the center pixel
    int u = msg->width / 2;
    int v = msg->height / 2;
    int W = 3.0;
    int H = 3.0;

    // Calculate the boundaries of the rectangular region
    int left = u - W / 2;
    int right = u + (W - 1) / 2;
    int top = v - H / 2;
    int bottom = v + (H - 1) / 2;

    // Calculate the total number of pixels in the rectangular region
    int numPixels = W * H;

    // Calculate the linear indices for the pixels within the rectangular region
    std::vector<int> pixelIndices;
    for (int y = top; y <= bottom; ++y)
    {
        for (int x = left; x <= right; ++x)
        {
            pixelIndices.push_back(x + msg->width * y);
        }
    }

    // Calculate the sum of depths within the rectangular region

    float sumDepths = 0.0;
    int validDepthsCount = 0;  // Count of valid depth values

    for (int idx : pixelIndices)
    {
        if (!std::isnan(depths[idx]) && !std::isinf(depths[idx])) {
            sumDepths += depths[idx];
            ++validDepthsCount;
            // Output the depth value of the current pixel
            RCLCPP_INFO(get_logger(), "Pixel depth : %g m", depths[idx]);
        }
    }

    if (validDepthsCount > 0) {
        float averageDepth = sumDepths / validDepthsCount;
        // Publish the average depth value
        std_msgs::msg::Float32 depth_info_msg;
        depth_info_msg.data = averageDepth;
        mDepthPub->publish(depth_info_msg);

        // Output the measure
        RCLCPP_INFO(get_logger(), "Average depth : %g m", depth_info_msg.data);
    } else {
        RCLCPP_WARN(get_logger(), "No valid depth values found.");
    }



    // Calculate the average depth
    // When doing the average, there might be some that are NaN values. If that is the case, 
    // ignore the NaN and do the average with the rest. 
    // float averageDepth = sumDepths / numPixels;

    // Publish the average depth value
    // std_msgs::msg::Float32 depth_info_msg;
    // depth_info_msg.data = averageDepth;
    //mDepthPub->publish(depth_info_msg);

    // Output the measure
    // RCLCPP_INFO(get_logger(), "Average depth : %g m", depth_info_msg.data);
    //RCLCPP_INFO(get_logger(), "Average depth : %g m | Sum Depths : %g m | Num Pixels : %d", averageDepth, sumDepths, numPixels);
  }


private:
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr mDepthSub;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr mDepthPub;  // Declare the publisher
};

// The main function
int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);

  auto depth_node = std::make_shared<MinimalDepthSubscriber>();

  rclcpp::spin(depth_node);
  rclcpp::shutdown();
  return 0;
}